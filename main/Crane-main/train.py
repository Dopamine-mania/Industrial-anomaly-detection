import models
from models import Crane
from models.prompt_ensemble import PromptLearner
from models.feature_refinement import FeatureRefinementModule
from dataset.dataset import Dataset
from __init__ import DATASETS_ROOT

from utils.transform import get_transform
from utils.loss import FocalLoss, BinaryDiceLoss
from utils.logger import get_logger
from utils.similarity import calc_similarity_logits, regrid_upsample
from utils.patch_graph import smooth_patch_tokens
from utils import (
    setup_seed,
    seed_worker,
    turn_gradient_off,
    str2bool,
    prepare_encode_image_module,
    precompute_image_features,
    CustomTensorDataset
)

import sys
import os
import argparse
import subprocess
import math
import re

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

# import torch.profiler

def _maybe_reserve_vram(args, device: str):
    """
    Optional VRAM reservation to satisfy operational requirements.
    This does NOT speed up training; it only increases GPU memory occupancy.
    """
    reserve_frac = float(getattr(args, "vram_reserve_frac", 0.0) or 0.0)
    reserve_gb = float(getattr(args, "vram_reserve_gb", 0.0) or 0.0)
    if device != "cuda" or not torch.cuda.is_available():
        return None
    if reserve_frac <= 0.0 and reserve_gb <= 0.0:
        return None

    # Determine target bytes to be USED (not to allocate).
    props = torch.cuda.get_device_properties(0)
    total_bytes = int(props.total_memory)
    if reserve_gb > 0.0:
        target_used = int(reserve_gb * (1024**3))
    else:
        target_used = int(total_bytes * reserve_frac)
    target_used = max(0, min(target_used, total_bytes))

    # Current usage from CUDA API (free/total).
    free_bytes, total_bytes2 = torch.cuda.mem_get_info()
    total_bytes = int(total_bytes2)
    used_bytes = total_bytes - int(free_bytes)
    need_bytes = target_used - used_bytes
    if need_bytes <= 0:
        return None

    # Allocate fp16 to reduce overhead. Use chunked allocations to avoid
    # huge single tensors (can hit internal size limits and crash).
    chunk_mb = int(getattr(args, "vram_reserve_chunk_mb", 256) or 256)
    chunk_bytes = max(16 * 1024**2, chunk_mb * 1024**2)
    elem_bytes = 2

    reserved = []
    remaining = need_bytes
    while remaining > 0:
        alloc_bytes = min(chunk_bytes, remaining)
        n_elems = max(1, alloc_bytes // elem_bytes)
        try:
            reserved.append(torch.empty((n_elems,), dtype=torch.float16, device="cuda"))
            remaining -= n_elems * elem_bytes
        except Exception:
            # back off chunk size; stop if too small to make progress
            if chunk_bytes <= 16 * 1024**2:
                break
            chunk_bytes = int(chunk_bytes * 0.5)

    return reserved if reserved else None

def _aggregate_text_features_banks(text_pos, text_neg, batch_size, normal_num, anormaly_num):
    if normal_num > 1:
        text_pos = text_pos.view(batch_size, normal_num, -1).mean(dim=1)
    if anormaly_num > 1:
        text_neg = text_neg.view(batch_size, anormaly_num, -1).mean(dim=1)
    return torch.stack([text_pos, text_neg], dim=1)


def _encode_text_learn_chunked(
    model,
    prompts: torch.Tensor,
    tokenized_prompts: torch.Tensor,
    compound_prompts_text,
    chunk_size: int,
):
    """
    Chunked wrapper around model.encode_text_learn to reduce VRAM spikes when encoding many prompts.
    prompts: (N, 77, C), tokenized_prompts: (N, 77)
    """
    n = int(prompts.shape[0])
    if chunk_size is None or int(chunk_size) <= 0 or int(chunk_size) >= n:
        return model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text)
    outs = []
    cs = int(chunk_size)
    for s in range(0, n, cs):
        e = min(n, s + cs)
        outs.append(model.encode_text_learn(prompts[s:e], tokenized_prompts[s:e], compound_prompts_text))
    return torch.cat(outs, dim=0)


def _encode_text_features_mc(model, prompt_learner, image_features, args):
    if getattr(args, "use_bayes_prompt", False):
        n_samples = int(getattr(args, "bayes_num_samples_train", 1))
    else:
        n_samples = 1
    text_encode_chunk_size = int(getattr(args, "text_encode_chunk_size", 0) or 0)
    total = None
    kl_total = None
    for _ in range(n_samples):
        prompts, tokenized_prompts, compound_prompts_text, is_train_with_img_cls = prompt_learner(img_emb=image_features)

        if is_train_with_img_cls:
            tf_pos = _encode_text_learn_chunked(
                model,
                prompts[0],
                tokenized_prompts[0],
                compound_prompts_text,
                chunk_size=text_encode_chunk_size,
            ).float()
            tf_neg = _encode_text_learn_chunked(
                model,
                prompts[1],
                tokenized_prompts[1],
                compound_prompts_text,
                chunk_size=text_encode_chunk_size,
            ).float()
            tf = _aggregate_text_features_banks(
                tf_pos,
                tf_neg,
                batch_size=image_features.shape[0],
                normal_num=prompt_learner.normal_num,
                anormaly_num=prompt_learner.anormaly_num,
            )
        else:
            tf_all = _encode_text_learn_chunked(
                model,
                prompts,
                tokenized_prompts,
                compound_prompts_text,
                chunk_size=text_encode_chunk_size,
            ).float()
            tf_pos = tf_all[: prompt_learner.normal_num].mean(dim=0, keepdim=True)
            tf_neg = tf_all[prompt_learner.normal_num :].mean(dim=0, keepdim=True)
            tf = torch.stack([tf_pos.squeeze(0), tf_neg.squeeze(0)], dim=0).unsqueeze(0)

        tf = F.normalize(tf, dim=-1)
        total = tf if total is None else (total + tf)

        kl = prompt_learner.bayes_kl_loss()
        if kl is not None:
            kl_total = kl if kl_total is None else (kl_total + kl)

    tf = total / float(n_samples)
    kl_mean = (kl_total / float(n_samples)) if kl_total is not None else None
    return tf, kl_mean


def _find_latest_checkpoint(save_path: str):
    ckpt_dir = os.path.abspath(save_path)
    if not os.path.isdir(ckpt_dir):
        return None, 0
    best_epoch = 0
    best_path = None
    for name in os.listdir(ckpt_dir):
        if not (name.startswith("epoch_") and name.endswith(".pth")):
            continue
        try:
            epoch_num = int(name[len("epoch_") : -len(".pth")])
        except ValueError:
            continue
        if epoch_num > best_epoch:
            best_epoch = epoch_num
            best_path = os.path.join(ckpt_dir, name)
    return best_path, best_epoch


def _make_synthetic_anomaly_batch(images: torch.Tensor, args):
    """
    Create a synthetic "abnormal" view from normal-only images (no test leakage).
    Returns:
      syn_images: (B,3,H,W)
      syn_mask:   (B,1,H,W) binary mask of modified region (for patch supervision)
    """
    prob = float(getattr(args, "synthetic_anomaly_prob", 0.0) or 0.0)
    if prob <= 0.0:
        return None, None

    mode = str(getattr(args, "synthetic_anomaly_mode", "cutpaste"))
    noise_std = float(getattr(args, "synthetic_anomaly_noise_std", 0.2))
    area_min = float(getattr(args, "synthetic_anomaly_area_min", 0.02))
    area_max = float(getattr(args, "synthetic_anomaly_area_max", 0.15))
    aspect_min = float(getattr(args, "synthetic_anomaly_aspect_min", 0.3))
    aspect_max = float(getattr(args, "synthetic_anomaly_aspect_max", 3.0))

    B, C, H, W = images.shape
    syn = images.clone()
    mask = torch.zeros((B, 1, H, W), device=images.device, dtype=images.dtype)

    for i in range(B):
        if torch.rand((), device=images.device).item() > prob:
            continue

        area_ratio = float(torch.empty((), device=images.device).uniform_(area_min, area_max).item())
        aspect = float(torch.empty((), device=images.device).uniform_(aspect_min, aspect_max).item())

        rect_area = max(1.0, area_ratio * float(H * W))
        rect_h = int(round(math.sqrt(rect_area / aspect)))
        rect_w = int(round(math.sqrt(rect_area * aspect)))
        rect_h = max(1, min(rect_h, H))
        rect_w = max(1, min(rect_w, W))

        y0 = int(torch.randint(0, max(1, H - rect_h + 1), (1,), device=images.device).item())
        x0 = int(torch.randint(0, max(1, W - rect_w + 1), (1,), device=images.device).item())
        y1 = y0 + rect_h
        x1 = x0 + rect_w

        if mode == "gaussian":
            region = syn[i, :, y0:y1, x0:x1]
            syn[i, :, y0:y1, x0:x1] = region + noise_std * torch.randn_like(region)
        elif mode == "cutpaste":
            sy0 = int(torch.randint(0, max(1, H - rect_h + 1), (1,), device=images.device).item())
            sx0 = int(torch.randint(0, max(1, W - rect_w + 1), (1,), device=images.device).item())
            patch = syn[i, :, sy0 : sy0 + rect_h, sx0 : sx0 + rect_w].clone()
            syn[i, :, y0:y1, x0:x1] = patch
        else:
            raise ValueError(f"Unknown synthetic_anomaly_mode={mode!r}")

        mask[i, 0, y0:y1, x0:x1] = 1.0

    return syn, mask


def _patch_level_ce_loss(patch_features, text_features, labels, temp: float):
    """
    Patch-level CE loss on CLIP patch tokens.
    - patch_features: Tensor (B, L, C) or list[Tensor(B, L, C)]
    - text_features: Tensor (B, 2, C) or (1, 2, C)
    - labels: Tensor (B,)
    """
    if patch_features is None:
        return None
    if torch.is_tensor(patch_features):
        if patch_features.dim() == 4:
            # (N_layers, B, L, C)
            patch_list = [patch_features[i] for i in range(patch_features.shape[0])]
        else:
            patch_list = [patch_features]
    else:
        patch_list = list(patch_features)
    if len(patch_list) == 0:
        return None

    losses = []
    for pf in patch_list:
        if pf is None:
            continue
        # logits: (B, L, 2)
        patch_logits = calc_similarity_logits(pf, text_features, temp=temp)
        B, L, _ = patch_logits.shape
        if labels.dim() == 2:
            target = labels.to(device=patch_logits.device).long().reshape(-1)
        else:
            target = labels.view(B, 1).expand(B, L).reshape(-1)
        losses.append(F.cross_entropy(patch_logits.reshape(B * L, -1), target))
    if not losses:
        return None
    return torch.stack(losses).mean()


def _patch_level_constrained_similarity_loss(
    patch_features,
    text_features,
    labels,
    temp: float,
    pos_threshold: float,
    margin: float,
):
    """
    Constrained similarity loss (normal-only friendly).
    Enforces:
      - pos >= pos_threshold
      - pos - neg >= margin
    but only until constraints are satisfied (hinge), reducing "runaway" collapse.
    """
    if patch_features is None:
        return None
    if torch.is_tensor(patch_features):
        if patch_features.dim() == 4:
            patch_list = [patch_features[i] for i in range(patch_features.shape[0])]
        else:
            patch_list = [patch_features]
    else:
        patch_list = list(patch_features)
    if len(patch_list) == 0:
        return None

    losses = []
    for pf in patch_list:
        if pf is None:
            continue
        B, L, _C = pf.shape
        tf = text_features
        if tf.shape[0] == 1 and B != 1:
            tf = tf.expand(B, -1, -1)

        # cosine similarity (scaled by temp for comparability with logits)
        sim = torch.einsum("blc,bkc->blk", pf, tf) / float(temp)

        if labels.dim() == 2:
            lbl = labels.long().to(device=sim.device).view(B, L, 1)
        else:
            lbl = labels.long().view(B, 1, 1).expand(B, L, 1)
        pos = sim.gather(2, lbl).squeeze(2)
        neg = sim.gather(2, 1 - lbl).squeeze(2)

        loss_pos = F.relu(float(pos_threshold) - pos).mean()
        loss_margin = F.relu(float(margin) - (pos - neg)).mean()
        losses.append(loss_pos + loss_margin)

    if not losses:
        return None
    return torch.stack(losses).mean()


def train(args):
    logger = get_logger(args.save_path)

    preprocess, target_transform = get_transform(args)
    train_data = Dataset(roots=args.train_data_path, transform=preprocess, 
                        target_transform=target_transform, dataset_name=args.dataset, kwargs=args)
    g = torch.Generator()
    g.manual_seed(args.seed)
    # train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True) # More basic for FPS comparison
    num_workers = int(getattr(args, "num_workers", 4))
    prefetch_factor = int(getattr(args, "prefetch_factor", 2))
    dl_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=g,
        worker_init_fn=seed_worker,
    )
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = prefetch_factor
    train_dataloader = DataLoader(train_data, **dl_kwargs)  # Faster (and safer under RAM limits)
    print(f"Length of the dataset: {len(train_data)}")

    ##########################################################################################
    device = 'cuda' if torch.cuda.is_available() else "cpu"    
    print(device)

    crane_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx, 'others': args}    
    model, _ = models.load("ViT-L/14@336px", device=device, design_details = crane_parameters)
    model = turn_gradient_off(model) 
    model.visual.replace_with_EAttn(to_layer=20, type=args.attn_type) # Replace last 20 layers
    if args.dino_model != 'none':
        model.use_DAttn(args.dino_model)
        
    prompt_learner = PromptLearner(model.to("cpu"), crane_parameters)
    sbp = Crane.ScoreBasePooling()
    frm = None
    if getattr(args, "use_feature_refinement_module", False):
        dim = model.ln_final.weight.shape[0]
        frm = FeatureRefinementModule(
            dim=dim,
            mode=getattr(args, "frm_type", "scalar"),
            alpha_init=float(getattr(args, "frm_alpha_init", 0.0)),
        )

    model.to(device)
    prompt_learner.to(device)
    if frm is not None:
        frm.to(device)

    # Prompt training scope: reduce drift to preserve zero-shot manifold.
    prompt_train_mode = str(getattr(args, "prompt_train_mode", "all"))
    if prompt_train_mode in {"bayes_only", "residual_only"}:
        prompt_learner.ctx_pos.requires_grad_(False)
        prompt_learner.ctx_neg.requires_grad_(False)
        for p in getattr(prompt_learner, "compound_prompts_text", []):
            p.requires_grad_(False)
    if prompt_train_mode == "residual_only":
        # additionally freeze Bayes modules except the residual alpha gates
        for name, p in prompt_learner.named_parameters():
            if name in {"bayes_residual_alpha_pos", "bayes_residual_alpha_neg"}:
                continue
            if not (name.startswith("ctx_pos") or name.startswith("ctx_neg") or name.startswith("compound_prompts_text")):
                p.requires_grad_(False)

    # Optional VRAM reservation (cosmetic; does not accelerate training).
    _vram_reservation = _maybe_reserve_vram(args, device)

    # Auto-resume (best-effort): reload prompt_learner (+ optional FRM) from the latest epoch_*.pth
    start_epoch = 0
    if getattr(args, "auto_resume", True) or getattr(args, "resume_path", None):
        ckpt_path = getattr(args, "resume_path", None)
        ckpt_epoch = 0
        if ckpt_path is None:
            ckpt_path, ckpt_epoch = _find_latest_checkpoint(args.save_path)
        else:
            ckpt_path = os.path.abspath(ckpt_path)
            m = re.search(r"epoch_(\d+)\.pth$", os.path.basename(ckpt_path))
            if m:
                ckpt_epoch = int(m.group(1))
            else:
                ckpt_epoch = int(getattr(args, "resume_epoch", 0) or 0)

        if ckpt_path is not None and os.path.isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            prompt_learner.load_state_dict(checkpoint["prompt_learner"], strict=True)
            if frm is not None and "feature_refinement_module" in checkpoint:
                frm.load_state_dict(checkpoint["feature_refinement_module"], strict=True)
            start_epoch = max(0, ckpt_epoch)
            logger.info(f"Auto-resume from {ckpt_path} (epoch {start_epoch})")

    ##########################################################################################
    params = list(prompt_learner.parameters())
    if frm is not None:
        params += list(frm.parameters())
    optimizer = torch.optim.Adam(
        params,
        lr=args.learning_rate,
        betas=(0.6, 0.999),
        weight_decay=float(getattr(args, "weight_decay", 0.0)),
    )
    
    precompute = False
    if precompute:
        encode_image_module = prepare_encode_image_module(model, args.features_list)
        precompute_features, pathes = precompute_image_features(train_data, encode_image_module, args)
        precompute_dataset = CustomTensorDataset(precompute_features, pathes)
        train_dataloader = DataLoader(precompute_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                       generator=g, worker_init_fn=seed_worker)
        model.visual.to('cpu')
    
    # losses
    ce_loss_focal = FocalLoss() 
    loss_dice = BinaryDiceLoss()

    model.eval()
    prompt_learner.train()
    for epoch in tqdm(range(start_epoch, args.epoch)):
        loss_list = []

        with tqdm(train_dataloader) as batch_tqdm:
            for items in batch_tqdm:
                label =  items['anomaly'].to(device)
                abnorm_mask = items['abnorm_mask'].squeeze().to(device)
                
                if precompute:
                    image_features, patch_features = items['image_features'].to(device), items['patch_features'].to(device)
                    patch_features = patch_features.permute(1, 0, *range(2, patch_features.dim())) # 4, N, L, C
                    mask_for_patches = None
                    label_img = label
                else:
                    image = items['img'].to(device)
                    # Optional: synthetic anomalies (normal-only compliant) to create pseudo-negatives.
                    # Only used in the Bayes-aligned branch (otherwise it would conflict with seg losses).
                    use_bayes_aligned = bool(getattr(args, "use_bayes_prompt", False) and getattr(args, "bayes_align_official", True))
                    syn_img = syn_mask = None
                    if use_bayes_aligned and float(getattr(args, "synthetic_anomaly_prob", 0.0) or 0.0) > 0.0:
                        syn_img, syn_mask = _make_synthetic_anomaly_batch(image, args)

                    if syn_img is not None and syn_mask is not None:
                        # Use the synthesized view directly (same batch size).
                        # Patch supervision uses syn_mask (1 => pseudo-abnormal patch).
                        image_for_encode = syn_img
                        mask_for_patches = syn_mask
                        label_img = label
                    else:
                        image_for_encode = image
                        mask_for_patches = None
                        label_img = label

                    # Bayes-PFL-aligned training: request patch tokens to keep DINO DAttn active and to enable
                    # patch-level supervision (spatial constraints).
                    image_features, patch_features = model.encode_image(
                        image_for_encode,
                        args.features_list,
                        self_cor_attn_layers=20,
                        vfm_num_layers=int(getattr(args, "vfm_num_layers", 1)),
                    )
                    # patch_features = torch.stack(patch_features, dim=0) 
                if frm is not None:
                    image_features = frm(image_features)
                    if isinstance(patch_features, list):
                        patch_features = [frm(pf) for pf in patch_features]
                    elif torch.is_tensor(patch_features):
                        patch_features = frm(patch_features)

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                if isinstance(patch_features, list):
                    if len(patch_features) > 0:
                        patch_features = [pf / pf.norm(dim=-1, keepdim=True) for pf in patch_features]
                        patch_features = torch.stack(patch_features, dim=0)
                    else:
                        patch_features = None
                elif torch.is_tensor(patch_features):
                    patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)

                # Optional: patch graph smoothing (neighbor averaging on patch grid).
                if getattr(args, "patch_graph_smooth", False) and patch_features is not None:
                    lam = float(getattr(args, "patch_graph_lambda", 0.1))
                    if torch.is_tensor(patch_features) and patch_features.dim() == 4:
                        patch_features = torch.stack(
                            [smooth_patch_tokens(patch_features[i], lam) for i in range(patch_features.shape[0])],
                            dim=0,
                        )
                    elif torch.is_tensor(patch_features) and patch_features.dim() == 3:
                        patch_features = smooth_patch_tokens(patch_features, lam)
            
                # Text Features
                #########################################################################
                text_features, kl_loss = _encode_text_features_mc(model, prompt_learner, image_features, args)
                text_features = text_features.float()  # (B,2,C) or (1,2,C)

                # Bayes-PFL aligned objective: image-level contrast (+ optional PFL regularizer).
                if getattr(args, "use_bayes_prompt", False) and getattr(args, "bayes_align_official", True):
                    patch_labels = None
                    if mask_for_patches is not None and patch_features is not None:
                        try:
                            if torch.is_tensor(patch_features) and patch_features.dim() == 4:
                                L = int(patch_features.shape[2])
                            elif torch.is_tensor(patch_features) and patch_features.dim() == 3:
                                L = int(patch_features.shape[1])
                            else:
                                L = None
                            if L is not None:
                                g = int(math.isqrt(L))
                                if g * g == L:
                                    pooled = F.adaptive_max_pool2d(mask_for_patches, (g, g)).flatten(1)
                                    patch_labels = (pooled > 0).to(dtype=torch.long)
                        except Exception:
                            patch_labels = None

                    patch_ce_temp = float(getattr(args, "bayes_patch_ce_temp", 0.07))
                    patch_loss_mode = str(getattr(args, "bayes_patch_loss", "ce"))
                    if patch_loss_mode == "ce":
                        patch_loss = _patch_level_ce_loss(
                            patch_features,
                            text_features,
                            labels=(patch_labels if patch_labels is not None else label_img.long().to(device)),
                            temp=patch_ce_temp,
                        )
                    elif patch_loss_mode == "constrained":
                        patch_loss = _patch_level_constrained_similarity_loss(
                            patch_features,
                            text_features,
                            labels=(patch_labels if patch_labels is not None else label_img.long().to(device)),
                            temp=patch_ce_temp,
                            pos_threshold=float(getattr(args, "bayes_constrained_pos_th", 0.2)),
                            margin=float(getattr(args, "bayes_constrained_margin", 0.1)),
                        )
                    else:
                        raise ValueError(f"Unknown bayes_patch_loss={patch_loss_mode!r}")

                    if patch_loss is None:
                        patch_loss = torch.zeros([], device=device)

                    pfl_loss = prompt_learner.bayes_pfl_loss()
                    if pfl_loss is None:
                        pfl_loss = torch.zeros([], device=device)

                    kl_w = float(getattr(args, "bayes_kl_weight", 0.0))
                    if kl_loss is None:
                        kl_loss_term = torch.zeros([], device=device)
                    else:
                        kl_loss_term = kl_loss.to(device=device)

                    img_w = float(getattr(args, "bayes_img_ce_weight", 0.2))
                    if img_w > 0:
                        image_logits = calc_similarity_logits(image_features, text_features, temp=0.01)
                        ce_img2txt_loss = F.cross_entropy(image_logits, label_img.long().to(device))
                    else:
                        ce_img2txt_loss = torch.zeros([], device=device)

                    ls = img_w * ce_img2txt_loss
                    ls = ls + float(getattr(args, "bayes_pfl_weight", 1.0)) * pfl_loss
                    ls = ls + float(getattr(args, "bayes_patch_ce_weight", 1.0)) * patch_loss
                    if kl_w > 0:
                        ls = ls + kl_w * kl_loss_term

                    optimizer.zero_grad()
                    ls.backward()
                    optimizer.step()

                    loss_list.append((0.0, 0.0, ce_img2txt_loss.item()))
                    batch_tqdm.set_description(
                        f"ce_fc_ls: {0.0:.3f}, bcd_dice_ls: {0.0:.3f}, ce_img_ls: {ce_img2txt_loss:.3f}, patch_ls: {patch_loss:.3f}"
                    )
                    continue

                # Similarity Map - Segmentation
                #########################################################################
                similarity_map_list = []
                for patch_feature in patch_features: 
                    pixel_logits = calc_similarity_logits(patch_feature, text_features, temp=0.07)
                    pixel_scores = pixel_logits.softmax(dim=-1)
                    similarity_map = regrid_upsample(pixel_scores, args.image_size, mode=args.interpolation) 
                    similarity_map_list.append((similarity_map, pixel_logits))

                ce_focal_loss = 0
                dice_loss = 0
                for i in range(len(similarity_map_list)):
                    whole_map = (1-similarity_map_list[i][0][...,0] + similarity_map_list[i][0][...,1])/2
                    smlr_map = similarity_map_list[i][0].permute(0, 3, 1, 2) 
 
                    dice_loss += loss_dice(whole_map, abnorm_mask)
                    ce_focal_loss += ce_loss_focal(smlr_map, abnorm_mask)
                    
                # Similarity Score - Classification
                #########################################################################
                if args.use_scorebase_pooling: 
                    alpha = 0.5
                    sms = [sm_lst[1] for sm_lst in similarity_map_list]
                    clustered_feature = sbp.forward(patch_features, sms) 
                    image_features = alpha * clustered_feature + (1 - alpha) * image_features # aggregates the class token and the clustered features for more comprehensive information
                    image_features = F.normalize(image_features, dim=1)

                image_logits = calc_similarity_logits(image_features, text_features, temp=0.01) # batch_size, 1, 768 @ batch_size, 768, 2 or 3
                ce_img2txt_loss = F.cross_entropy(image_logits, label.long().to(device)) 
                # txt2img_lbl = torch.stack([(1-label), label], dim=0)/label.sum()
                # ce_txt2img_loss = F.cross_entropy(image_logits.permute(1, 0), txt2img_lbl.to(device))                                         

                #loss
                optimizer.zero_grad()
                dice_loss *= 2
                ce_focal_loss *= 2
                ls = ce_focal_loss + dice_loss + 0.2 * ce_img2txt_loss
                if (
                    getattr(args, "use_bayes_prompt", False)
                    and float(getattr(args, "bayes_kl_weight", 0.0)) > 0
                    and kl_loss is not None
                ):
                    ls = ls + float(args.bayes_kl_weight) * kl_loss
                ls.backward() 
                optimizer.step()

                loss_list.append((ce_focal_loss.item(), dice_loss.item(), ce_img2txt_loss.item()))
                batch_tqdm.set_description(f"ce_fc_ls: {ce_focal_loss:.3f}, bcd_dice_ls: {dice_loss:.3f}, ce_img_ls: {ce_img2txt_loss:.3f}")                
        # logs
        ce_focal_ls, dice_ls, ce_img_ls = np.mean(loss_list, axis=0)
        log_template = 'epoch [{}/{}], ce_fc_ls:{:.4f}, bdc_ls:{:.4f}, ce_img_ls:{:.4f}'
        logger.info(log_template.format(epoch + 1, args.epoch, ce_focal_ls, dice_ls, ce_img_ls))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            prmtp_ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
            checkpoint_data = {"prompt_learner": prompt_learner.state_dict()}
            if frm is not None:
                checkpoint_data["feature_refinement_module"] = frm.state_dict()
            torch.save(checkpoint_data, prmtp_ckp_path)

if __name__ == '__main__':
    dss = ['mvtec']
    
    parser = argparse.ArgumentParser("Crane", add_help=True)
    parser.add_argument("--datasets_root_dir", type=str, default=f"{DATASETS_ROOT}")
    parser.add_argument("--train_data_path", type=str, nargs="+", default=[f"{DATASETS_ROOT}/{ds}/" for ds in dss])
    parser.add_argument("--save_path", type=str, default='./checkpoints/')
    parser.add_argument("--model_name", type=str, default="default") # NOTE: The "trained_on_<DATASET_NAME>" will be prepended to the model name for saving checkpoints
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")

    parser.add_argument("--type", type=str, default='train') 
    parser.add_argument("--device", type=int, default=0, help="cuda device")
    parser.add_argument("--epoch", type=int, default=5, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Adam weight decay")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="dataloader workers (reduce if RAM-limited)")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="dataloader prefetch factor (only if num_workers>0)")
    parser.add_argument(
        "--text_encode_chunk_size",
        type=int,
        default=0,
        help="chunk size for text prompt encoding (0 disables; use to avoid CUDA OOM when batch/prompts are large)",
    )
    parser.add_argument("--aug_rate", type=float, default=0.0, help="augmentation rate")
    parser.add_argument("--vram_reserve_frac", type=float, default=0.0, help="reserve GPU VRAM to this used fraction (0 disables)")
    parser.add_argument("--vram_reserve_gb", type=float, default=0.0, help="reserve GPU VRAM to this used GB (0 disables)")
    parser.add_argument("--vram_reserve_chunk_mb", type=int, default=256, help="VRAM reserve allocation chunk size (MB)")
    parser.add_argument("--train_good_only", type=str2bool, default=True, help="train split normal-only (anti-leakage)")
    parser.add_argument("--auto_resume", type=str2bool, default=True, help="auto resume from latest checkpoint")
    parser.add_argument("--resume_path", type=str, default=None, help="explicit checkpoint path to resume from")
    parser.add_argument("--resume_epoch", type=int, default=0, help="used if resume_path has no epoch_XX suffix")

    parser.add_argument("--dataset", type=str, nargs="+", default=[f'{ds}' for ds in dss], help="train dataset name")
    parser.add_argument("--target_class", type=str, default=None, help="optional: restrict to specific class (e.g. bottle)")
    parser.add_argument("--k_shot", type=int, default=0, help="samples per class for few-shot learning. 0 means use all data.")
    parser.add_argument("--portion", type=float, default=1) 

    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--features_list", type=int, nargs="+", default=[24], help="layer features used")
    parser.add_argument("--interpolation", type=str, choices=['nearest', 'bilinear'], default='nearest') 

    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
     
    parser.add_argument("--train_with_img_cls_prob", type=float, default=1)
    parser.add_argument("--train_with_img_cls_type", type=str, choices=["none", "replace_prefix", "replace_suffix", "pad_prefix", "pad_suffix"], default="pad_suffix")
    parser.add_argument(
        "--ctx_init",
        type=str,
        choices=["random", "zeros", "clip"],
        default="random",
        help="init for ctx_pos/ctx_neg prompt context vectors",
    )
    parser.add_argument(
        "--ctx_init_phrase",
        type=str,
        default="a photo of a",
        help="when ctx_init=clip, initialize context vectors from this phrase's token embeddings",
    )
    parser.add_argument("--dino_model", type=str, choices=['none', 'dinov2', 'dino', 'sam'], default='dinov2')
    parser.add_argument("--vfm_num_layers", type=int, default=1, help="number of VFM intermediate layers to use inside the visual encoder (for DAttn).")
    parser.add_argument("--both_eattn_dattn", type=str2bool, default=True)
    parser.add_argument("--use_scorebase_pooling", type=str2bool, default=True) 
    parser.add_argument("--attn_type", type=str, choices=["vv", "kk", "qq", "qq+kk", "qq+kk+vv", "(q+k+v)^2"], default="qq+kk+vv")
    parser.add_argument(
        "--prompt_train_mode",
        type=str,
        choices=["all", "bayes_only", "residual_only"],
        default="all",
        help="which PromptLearner params to train (reduces drift vs zero-shot)",
    )

    # Bayes-PFL (text-side plugin)
    parser.add_argument("--use_bayes_prompt", type=str2bool, default=False)
    parser.add_argument("--bayes_num_samples", type=int, default=8)
    parser.add_argument("--bayes_flow_steps", type=int, default=4)
    parser.add_argument("--bayes_flow_type", type=str, choices=["planar", "residual"], default="planar")
    parser.add_argument("--bayes_kl_weight", type=float, default=0.01)
    parser.add_argument("--bayes_condition_on_image", type=str2bool, default=True)
    parser.add_argument("--bayes_init_logstd", type=float, default=math.log(0.02))
    parser.add_argument("--bayes_use_residual", type=str2bool, default=True, help="residual-gate Bayes prompt updates")
    parser.add_argument("--bayes_residual_alpha_init", type=float, default=0.01, help="init alpha for residual-gated Bayes prompts")
    parser.add_argument("--bayes_align_official", type=str2bool, default=True)
    parser.add_argument("--bayes_num_samples_train", type=int, default=1)
    parser.add_argument("--bayes_pfl_weight", type=float, default=1.0)
    parser.add_argument("--bayes_img_ce_weight", type=float, default=0.2)
    parser.add_argument("--bayes_patch_ce_weight", type=float, default=1.0, help="patch-level CE weight (Bayes aligned training)")
    parser.add_argument("--bayes_patch_ce_temp", type=float, default=0.07, help="patch-level CE temperature")
    parser.add_argument("--bayes_patch_loss", type=str, choices=["ce", "constrained"], default="ce", help="patch loss for Bayes aligned training")
    parser.add_argument("--bayes_constrained_pos_th", type=float, default=0.2, help="constrained loss: min similarity threshold for positives")
    parser.add_argument("--bayes_constrained_margin", type=float, default=0.1, help="constrained loss: pos-neg margin")
    parser.add_argument("--patch_graph_smooth", type=str2bool, default=False, help="apply neighbor smoothing on patch tokens")
    parser.add_argument("--patch_graph_lambda", type=float, default=0.1, help="smoothing strength for patch_graph_smooth")
    parser.add_argument("--synthetic_anomaly_prob", type=float, default=0.0, help="probability to synthesize anomalies from normal images")
    parser.add_argument("--synthetic_anomaly_mode", type=str, choices=["cutpaste", "gaussian"], default="cutpaste", help="synthetic anomaly type")
    parser.add_argument("--synthetic_anomaly_noise_std", type=float, default=0.2, help="std for gaussian synthetic anomalies")
    parser.add_argument("--synthetic_anomaly_area_min", type=float, default=0.02, help="min rectangle area ratio for synthetic anomalies")
    parser.add_argument("--synthetic_anomaly_area_max", type=float, default=0.15, help="max rectangle area ratio for synthetic anomalies")
    parser.add_argument("--synthetic_anomaly_aspect_min", type=float, default=0.3, help="min aspect ratio (w/h) for synthetic anomalies")
    parser.add_argument("--synthetic_anomaly_aspect_max", type=float, default=3.0, help="max aspect ratio (w/h) for synthetic anomalies")

    # Feature refinement (attention-ish, minimal-risk)
    parser.add_argument("--use_feature_refinement_module", type=str2bool, default=False)
    parser.add_argument("--frm_type", type=str, choices=["scalar", "linear"], default="scalar")
    parser.add_argument("--frm_alpha_init", type=float, default=0.0)
    parser.add_argument("--why", type=str, default="Neccessity of the experiment")

    args = parser.parse_args()
    
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device) 
        command = [sys.executable, __file__, ] + sys.argv[1:]  
        process = subprocess.Popen(command, env=os.environ)
        process.wait()
        
    else:
        setup_seed(args.seed)        

        # paths
        args.train_data_path = [f"{args.datasets_root_dir}/{ds}/" for ds in args.dataset]
        args.save_path = f'{args.save_path}/trained_on_{"_".join(args.dataset)}_{args.model_name}/'
        print(f'running {args.model_name}')

        train(args)
