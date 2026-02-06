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


def _encode_text_features_mc(model, prompt_learner, image_features, args):
    if getattr(args, "use_bayes_prompt", False):
        n_samples = int(getattr(args, "bayes_num_samples_train", 1))
    else:
        n_samples = 1
    total = None
    kl_total = None
    for _ in range(n_samples):
        prompts, tokenized_prompts, compound_prompts_text, is_train_with_img_cls = prompt_learner(img_emb=image_features)

        if is_train_with_img_cls:
            tf_pos = model.encode_text_learn(prompts[0], tokenized_prompts[0], compound_prompts_text).float()
            tf_neg = model.encode_text_learn(prompts[1], tokenized_prompts[1], compound_prompts_text).float()
            tf = _aggregate_text_features_banks(
                tf_pos,
                tf_neg,
                batch_size=image_features.shape[0],
                normal_num=prompt_learner.normal_num,
                anormaly_num=prompt_learner.anormaly_num,
            )
        else:
            tf_all = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
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
        target = labels.view(B, 1).expand(B, L).reshape(-1)
        losses.append(F.cross_entropy(patch_logits.reshape(B * L, -1), target))
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

    # Optional VRAM reservation (cosmetic; does not accelerate training).
    _vram_reservation = _maybe_reserve_vram(args, device)

    # Auto-resume (best-effort): reload prompt_learner (+ optional FRM) from the latest epoch_*.pth
    start_epoch = 0
    if getattr(args, "auto_resume", True) or getattr(args, "resume_path", None):
        ckpt_path = getattr(args, "resume_path", None)
        ckpt_epoch = 0
        if ckpt_path is None:
            ckpt_path, ckpt_epoch = _find_latest_checkpoint(args.save_path)
        if ckpt_path is not None and ckpt_epoch > 0:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            prompt_learner.load_state_dict(checkpoint["prompt_learner"], strict=True)
            if frm is not None and "feature_refinement_module" in checkpoint:
                frm.load_state_dict(checkpoint["feature_refinement_module"], strict=True)
            start_epoch = ckpt_epoch
            logger.info(f"Auto-resume from {ckpt_path} (epoch {ckpt_epoch})")

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
                else:
                    image = items['img'].to(device)
                    # Bayes-PFL-aligned training: we still request patch tokens to:
                    # (1) keep DINO DAttn active (it is computed only for out_layers),
                    # (2) enable patch-level supervision (spatial constraint).
                    image_features, patch_features = model.encode_image(image, args.features_list, self_cor_attn_layers=20)
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
            
                # Text Features
                #########################################################################
                text_features, kl_loss = _encode_text_features_mc(model, prompt_learner, image_features, args)
                text_features = text_features.float()  # (B,2,C) or (1,2,C)

                # Bayes-PFL aligned objective: image-level contrast (+ optional PFL regularizer).
                if getattr(args, "use_bayes_prompt", False) and getattr(args, "bayes_align_official", True):
                    image_logits = calc_similarity_logits(image_features, text_features, temp=0.01)
                    ce_img2txt_loss = F.cross_entropy(image_logits, label.long().to(device))

                    patch_ce_temp = float(getattr(args, "bayes_patch_ce_temp", 0.07))
                    patch_ce_loss = _patch_level_ce_loss(
                        patch_features,
                        text_features,
                        labels=label.long().to(device),
                        temp=patch_ce_temp,
                    )
                    if patch_ce_loss is None:
                        patch_ce_loss = torch.zeros([], device=device)

                    pfl_loss = prompt_learner.bayes_pfl_loss()
                    if pfl_loss is None:
                        pfl_loss = torch.zeros([], device=device)

                    ls = float(getattr(args, "bayes_img_ce_weight", 0.2)) * ce_img2txt_loss + float(
                        getattr(args, "bayes_pfl_weight", 1.0)
                    ) * pfl_loss
                    ls = ls + float(getattr(args, "bayes_patch_ce_weight", 1.0)) * patch_ce_loss

                    optimizer.zero_grad()
                    ls.backward()
                    optimizer.step()

                    loss_list.append((0.0, 0.0, ce_img2txt_loss.item()))
                    batch_tqdm.set_description(
                        f"ce_fc_ls: {0.0:.3f}, bcd_dice_ls: {0.0:.3f}, ce_img_ls: {ce_img2txt_loss:.3f}, ce_patch_ls: {patch_ce_loss:.3f}"
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
    parser.add_argument("--aug_rate", type=float, default=0.0, help="augmentation rate")
    parser.add_argument("--vram_reserve_frac", type=float, default=0.0, help="reserve GPU VRAM to this used fraction (0 disables)")
    parser.add_argument("--vram_reserve_gb", type=float, default=0.0, help="reserve GPU VRAM to this used GB (0 disables)")
    parser.add_argument("--vram_reserve_chunk_mb", type=int, default=256, help="VRAM reserve allocation chunk size (MB)")
    parser.add_argument("--train_good_only", type=str2bool, default=True, help="train split normal-only (anti-leakage)")
    parser.add_argument("--auto_resume", type=str2bool, default=True, help="auto resume from latest checkpoint")
    parser.add_argument("--resume_path", type=str, default=None, help="explicit checkpoint path to resume from")

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
    parser.add_argument("--dino_model", type=str, choices=['none', 'dinov2', 'dino', 'sam'], default='dinov2')
    parser.add_argument("--both_eattn_dattn", type=str2bool, default=True)
    parser.add_argument("--use_scorebase_pooling", type=str2bool, default=True) 
    parser.add_argument("--attn_type", type=str, choices=["vv", "kk", "qq", "qq+kk", "qq+kk+vv", "(q+k+v)^2"], default="qq+kk+vv")

    # Bayes-PFL (text-side plugin)
    parser.add_argument("--use_bayes_prompt", type=str2bool, default=False)
    parser.add_argument("--bayes_num_samples", type=int, default=8)
    parser.add_argument("--bayes_flow_steps", type=int, default=4)
    parser.add_argument("--bayes_flow_type", type=str, choices=["planar", "residual"], default="planar")
    parser.add_argument("--bayes_kl_weight", type=float, default=0.01)
    parser.add_argument("--bayes_condition_on_image", type=str2bool, default=True)
    parser.add_argument("--bayes_init_logstd", type=float, default=math.log(0.02))
    parser.add_argument("--bayes_align_official", type=str2bool, default=True)
    parser.add_argument("--bayes_num_samples_train", type=int, default=1)
    parser.add_argument("--bayes_pfl_weight", type=float, default=1.0)
    parser.add_argument("--bayes_img_ce_weight", type=float, default=0.2)
    parser.add_argument("--bayes_patch_ce_weight", type=float, default=1.0, help="patch-level CE weight (Bayes aligned training)")
    parser.add_argument("--bayes_patch_ce_temp", type=float, default=0.07, help="patch-level CE temperature")

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
