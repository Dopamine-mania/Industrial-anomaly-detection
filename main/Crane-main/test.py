import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import models
from models import Crane
from models.prompt_ensemble import PromptLearner
from models.prompt_ensemble import tokenize as clip_tokenize
from models.feature_refinement import FeatureRefinementModule
from dataset.dataset import Dataset
from __init__ import DATASETS_ROOT

from utils.transform import get_transform
from utils.visualization import visualizer
from utils.metrics import image_level_metrics, pixel_level_metrics
from utils.logger import get_logger, save_args_to_file
from utils.similarity import calc_similarity_logits, regrid_upsample
from utils.patch_graph import smooth_patch_tokens
from utils import (
    setup_seed,
    turn_gradient_off,
    str2bool,
    make_human_readable_name,
)

from scipy.ndimage import gaussian_filter
import pandas as pd 
import sys
import os
import subprocess
import argparse
from tqdm import tqdm
from tabulate import tabulate
import math

from termcolor import colored

def _debug_print_similarity_samples(model, prompt_learner, dataloader, args, max_each: int):
    """
    Print raw cosine similarities to normal/abnormal prompts for a few normal and abnormal samples.
    This is a sanity check for the prompt directionality.
    """
    pos = 0
    neg = 0
    with torch.no_grad():
        for items in dataloader:
            imgs = items["img"].cuda()
            gt = items["anomaly"].cpu().numpy().tolist()
            img_paths = items.get("img_path", [""] * imgs.shape[0])

            image_features, _patch_list = model.encode_image(imgs, args.features_list, self_cor_attn_layers=20)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # use the same Bayes conditioning path as inference
            if getattr(args, "use_bayes_prompt", False) and getattr(args, "bayes_condition_on_image", True):
                text_features = _encode_text_features_mc(model, prompt_learner, image_features, args)
            elif args.train_with_img_cls_prob != 0:
                text_features = _encode_text_features_mc(model, prompt_learner, image_features, args)
            else:
                text_features = prompt_learner.text_features.to(imgs.device)  # type: ignore[attr-defined]

            if text_features.shape[0] == 1 and image_features.shape[0] != 1:
                text_features = text_features.expand(image_features.shape[0], -1, -1)

            # cosine similarities (unscaled)
            cos = torch.einsum("bc,bkc->bk", image_features, text_features)

            for i in range(imgs.shape[0]):
                is_abn = int(gt[i])
                if is_abn == 0 and pos >= max_each:
                    continue
                if is_abn == 1 and neg >= max_each:
                    continue
                s0 = float(cos[i, 0].detach().cpu())
                s1 = float(cos[i, 1].detach().cpu())
                print(f"[dbg-sim] y={is_abn} cos(normal)={s0:.4f} cos(abnormal)={s1:.4f} path={img_paths[i]}")
                if is_abn == 0:
                    pos += 1
                else:
                    neg += 1
                if pos >= max_each and neg >= max_each:
                    return

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
    n_samples = int(args.bayes_num_samples) if getattr(args, "use_bayes_prompt", False) else 1
    text_encode_chunk_size = int(getattr(args, "text_encode_chunk_size", 0) or 0)
    total = None
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

    return total / float(n_samples)


def _encode_text_features_mc_static(model, prompt_learner, args, device):
    n_samples = int(args.bayes_num_samples) if getattr(args, "use_bayes_prompt", False) else 1
    text_encode_chunk_size = int(getattr(args, "text_encode_chunk_size", 0) or 0)
    total = None
    for _ in range(n_samples):
        prompts, tokenized_prompts, compound_prompts_text, _ = prompt_learner()
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
        tf = F.normalize(tf, dim=-1).to(device)
        total = tf if total is None else (total + tf)
    return total / float(n_samples)

def _encode_text_prompt_banks_fixed(model, class_name: str, device: str = "cuda"):
    """
    WinCLIP-style fixed prompts: encode normal/abnormal prompt banks.
    Returns:
      tf_pos_bank: (P, C) normalized
      tf_neg_bank: (Q, C) normalized
    """
    from models.state_prompts import NORMAL_STATE_TEMPLATES, ABNORMAL_STATE_TEMPLATES

    prompts_pos = [t.format(class_name) for t in NORMAL_STATE_TEMPLATES]
    prompts_neg = [t.format(class_name) for t in ABNORMAL_STATE_TEMPLATES]
    tokens_pos = clip_tokenize(prompts_pos).to(device=device)
    tokens_neg = clip_tokenize(prompts_neg).to(device=device)

    # NOTE: Crane's text Transformer uses ResidualAttentionBlock_learnable_token blocks, which expect
    # list inputs. `encode_text()` (standard CLIP path) will fail due to this architectural change.
    # We therefore reuse `encode_text_learn()` with an empty deep prompt list to emulate standard CLIP.
    cast_dtype = model.transformer.get_cast_dtype()
    with torch.no_grad():
        emb_pos = model.token_embedding(tokens_pos).type(cast_dtype)
        emb_neg = model.token_embedding(tokens_neg).type(cast_dtype)

        tf_pos = model.encode_text_learn(emb_pos, tokens_pos, deep_compound_prompts_text=[]).float()
        tf_neg = model.encode_text_learn(emb_neg, tokens_neg, deep_compound_prompts_text=[]).float()
        tf_pos = F.normalize(tf_pos, dim=-1)
        tf_neg = F.normalize(tf_neg, dim=-1)
    return tf_pos, tf_neg


def _reduce_prompt_scores(scores: torch.Tensor, mode: str):
    if mode == "max":
        return scores.max(dim=-1).values
    if mode == "mean":
        return scores.mean(dim=-1)
    if mode == "logsumexp":
        return torch.logsumexp(scores, dim=-1)
    raise ValueError(f"Unknown fixed_prompts_reduce={mode!r}")


def _fixed_prompt_logits(features: torch.Tensor, tf_pos: torch.Tensor, tf_neg: torch.Tensor, temp: float, reduce: str):
    """
    features: (B,C) or (B,L,C)
    tf_pos/tf_neg: (P,C)/(Q,C)
    returns logits: (B,2) or (B,L,2)
    """
    if features.dim() == 2:
        pos = (features @ tf_pos.t()) / float(temp)
        neg = (features @ tf_neg.t()) / float(temp)
        pos_r = _reduce_prompt_scores(pos, reduce)
        neg_r = _reduce_prompt_scores(neg, reduce)
        return torch.stack([pos_r, neg_r], dim=-1)
    if features.dim() == 3:
        pos = torch.einsum("blc,pc->blp", features, tf_pos) / float(temp)
        neg = torch.einsum("blc,qc->blq", features, tf_neg) / float(temp)
        pos_r = _reduce_prompt_scores(pos, reduce)
        neg_r = _reduce_prompt_scores(neg, reduce)
        return torch.stack([pos_r, neg_r], dim=-1)
    raise ValueError(f"Unexpected features.dim()={features.dim()}")

class ScoreCalculator(nn.Module):
    def __init__(self, base_model, class_details, args,
                 prompt_learner, score_base_pooling, feature_refinement_module=None):
        super().__init__()
        self.model = base_model
        self.class_details = class_details
        self.args = args

        self.prompt_learner = prompt_learner
        self.sbp = score_base_pooling
        self.frm = feature_refinement_module
        self._cached_text = {}
        self.fixed_tf_pos_bank = None
        self.fixed_tf_neg_bank = None

    def forward(self, image):
        with torch.no_grad():
            use_vfm_fusion = bool(getattr(self.args, "vfm_fusion", False))
            if use_vfm_fusion:
                image_features, patch_list, vfm_feats_list = self.model.encode_image(
                    image,
                    self.args.features_list,
                    self_cor_attn_layers=20,
                    vfm_num_layers=int(getattr(self.args, "vfm_num_layers", 4)),
                    return_vfm_feats=True,
                )
            else:
                image_features, patch_list = self.model.encode_image(image, self.args.features_list, self_cor_attn_layers=20)
                vfm_feats_list = None
            # patch_features = torch.stack(patch_list, dim=0)
        if self.frm is not None:
            image_features = self.frm(image_features)
            patch_list = [self.frm(pf) for pf in patch_list]

        image_features = image_features / image_features.norm(dim=-1, keepdim=True) 
        patch_features = [patch_feature / patch_feature.norm(dim=-1, keepdim=True) for patch_feature in patch_list] # Note

        # Multi-scale fusion: fuse CLIP patch tokens with VFM (e.g., DINOv2) intermediate features.
        if bool(getattr(self.args, "vfm_fusion", False)) and vfm_feats_list is not None and len(vfm_feats_list) > 0:
            try:
                # Average intermediate layers in feature space.
                vfm_feat = torch.stack([vf for vf in vfm_feats_list], dim=0).mean(dim=0)  # (B,C,H,W)
                B, C, H, W = vfm_feat.shape
                vfm_tokens = vfm_feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
                vfm_tokens = vfm_tokens / vfm_tokens.norm(dim=-1, keepdim=True)

                fused = []
                weight = float(getattr(self.args, "vfm_fusion_weight", 1.0))
                mode = str(getattr(self.args, "vfm_fusion_mode", "concat_mean"))
                for pf in patch_features:
                    Bb, L, Cc = pf.shape
                    if Bb != B:
                        vfm_t = vfm_tokens[:Bb]
                    else:
                        vfm_t = vfm_tokens

                    # Align token grid if lengths differ.
                    if vfm_t.shape[1] != L:
                        g = int(math.isqrt(L))
                        if g * g == L and H == W:
                            vf = vfm_feat
                            if vf.shape[-1] != g or vf.shape[-2] != g:
                                vf = F.interpolate(vf, size=(g, g), mode="bilinear", align_corners=False)
                            vfm_t = vf.permute(0, 2, 3, 1).reshape(B, g * g, C)
                            vfm_t = vfm_t / vfm_t.norm(dim=-1, keepdim=True)
                        else:
                            vfm_t = vfm_t[:, :L, :]

                    if vfm_t.shape[-1] != Cc:
                        # For now, require same embed dim (dinov2_vitb14_reg matches 768).
                        vfm_t = vfm_t[..., :Cc]

                    if mode == "add":
                        f = pf + weight * vfm_t
                    elif mode == "concat_mean":
                        cat = torch.cat([pf, weight * vfm_t], dim=-1)  # (B,L,2C)
                        f = cat.view(Bb, L, 2, Cc).mean(dim=2)
                    else:
                        raise ValueError(f"Unknown vfm_fusion_mode={mode!r}")
                    f = f / f.norm(dim=-1, keepdim=True)
                    fused.append(f)
                patch_features = fused
            except Exception:
                # Best-effort: if VFM fusion fails for any reason, fall back to CLIP-only tokens.
                pass

        patch_features = torch.stack(patch_features, dim=0) 
        if getattr(self.args, "patch_graph_smooth", False):
            lam = float(getattr(self.args, "patch_graph_lambda", 0.1))
            patch_features = torch.stack([smooth_patch_tokens(pf, lam) for pf in patch_features], dim=0)
        # image_features = F.normalize(image_features, dim=-1)
        # patch_features = F.normalize(patch_features, dim=-1)
        
        fixed_prompts = bool(getattr(self.args, "fixed_prompts", False))
        if fixed_prompts:
            text_features = None
        elif getattr(self.args, "use_bayes_prompt", False) and getattr(self.args, "bayes_condition_on_image", True):
            text_features = _encode_text_features_mc(self.model, self.prompt_learner, image_features, self.args)
        elif self.args.train_with_img_cls_prob != 0:
            text_features = _encode_text_features_mc(self.model, self.prompt_learner, image_features, self.args)
        else:
            text_features = self.text_features.to(image.device)

        # Similarity Map - Segmentation
        #########################################################################
        pixel_logits_list = []
        fixed_reduce = str(getattr(self.args, "fixed_prompts_reduce", "max"))
        for patch_feature in patch_features:
            if fixed_prompts:
                pixel_logits = _fixed_prompt_logits(
                    patch_feature,
                    self.fixed_tf_pos_bank,
                    self.fixed_tf_neg_bank,
                    temp=0.07,
                    reduce=fixed_reduce,
                )
            else:
                pixel_logits = calc_similarity_logits(patch_feature, text_features)
            pixel_logits_list.append(pixel_logits)

        if self.args.soft_mean:    
            similarity_maps = [regrid_upsample(pl.softmax(dim=-1), args.image_size) for pl in pixel_logits_list]
            score_map = torch.stack(similarity_maps).mean(dim=0)
        else:
            logits_maps = [regrid_upsample(pl, args.image_size) for pl in pixel_logits_list]
            mean_logits_map = torch.stack(logits_maps).mean(dim=0)
            score_map = mean_logits_map.softmax(dim=-1)
        anomaly_map = score_map[..., 1]

        # Classification Score
        #########################################################################
        if self.args.use_scorebase_pooling:
            alpha = 0.5
            clustered_feature = self.sbp.forward(patch_features, pixel_logits_list)
            image_features = alpha * clustered_feature + (1 - alpha) * image_features
            image_features = F.normalize(image_features, dim=1)

        if fixed_prompts:
            image_logits = _fixed_prompt_logits(
                image_features,
                self.fixed_tf_pos_bank,
                self.fixed_tf_neg_bank,
                temp=0.07,
                reduce=fixed_reduce,
            )
        else:
            image_logits = calc_similarity_logits(image_features, text_features)
        image_pred = image_logits.softmax(dim=-1)
        anomaly_score = image_pred[:, 1].detach()

        if getattr(self.args, "invert_scores", False):
            anomaly_score = 1.0 - anomaly_score
            anomaly_map = 1.0 - anomaly_map

        return anomaly_score, anomaly_map 

def compute_metrics_for_object(obj, dataset_results):
    dataset_results[obj]['imgs_masks'] = torch.stack(dataset_results[obj]['imgs_masks'])
    dataset_results[obj]['anomaly_maps'] = torch.stack(dataset_results[obj]['anomaly_maps'])
    
    image_auroc = image_level_metrics(dataset_results, obj, "image-auroc")
    image_ap = image_level_metrics(dataset_results, obj, "image-ap")
    image_f1 = image_level_metrics(dataset_results, obj, "image-f1")

    pixel_auroc = pixel_level_metrics(dataset_results, obj, "pixel-auroc")
    pixel_ap = pixel_level_metrics(dataset_results, obj, "pixel-ap")
    pixel_aupro = pixel_level_metrics(dataset_results, obj, "pixel-aupro")
    pixel_f1 = pixel_level_metrics(dataset_results, obj, "pixel-f1")

    dataset_results[obj] = None
    return {
        "pixel_auroc": pixel_auroc,
        "pixel_ap": pixel_ap,
        "pixel_aupro": pixel_aupro,
        "pixel_f1": pixel_f1,
        "image_auroc": image_auroc,
        "image_ap": image_ap,
        "image_f1": image_f1,
    }

def process_dataset(model, dataloader, class_details, args): 
    Crane_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx, 'others': args}
    prompt_learner = PromptLearner(model, Crane_parameters)
    checkpoint = None
    if not getattr(args, "skip_checkpoint_load", False):
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        missing_keys, unexpected_keys = prompt_learner.load_state_dict(checkpoint["prompt_learner"], strict=True)
        assert len(missing_keys) == 0, f"Missing keys in state dict: {missing_keys}"
        assert len(unexpected_keys) == 0, f"Unexpected keys in state dict: {unexpected_keys}"
    prompt_learner = prompt_learner.cuda()
    
    score_base_pooling = Crane.ScoreBasePooling()
    frm = None
    if checkpoint is not None and "feature_refinement_module" in checkpoint:
        dim = model.ln_final.weight.shape[0]
        frm = FeatureRefinementModule(
            dim=dim,
            mode=getattr(args, "frm_type", "scalar"),
            alpha_init=float(getattr(args, "frm_alpha_init", 0.0)),
        )
        frm.load_state_dict(checkpoint["feature_refinement_module"], strict=True)
        frm = frm.cuda()

    score_calc = ScoreCalculator(
        model, class_details, args,
        prompt_learner=prompt_learner,
        score_base_pooling=score_base_pooling,
        feature_refinement_module=frm,
    )

    if getattr(args, "fixed_prompts", False):
        # Pick class name: if evaluating a single class, prefer that for prompt semantics.
        class_name = "object"
        if getattr(args, "fixed_prompt_classname", None):
            class_name = str(args.fixed_prompt_classname)
        elif getattr(args, "target_class", None) and "," not in str(args.target_class):
            class_name = str(args.target_class).strip()
        with torch.no_grad():
            tf_pos, tf_neg = _encode_text_prompt_banks_fixed(model, class_name, device="cuda")
            score_calc.fixed_tf_pos_bank = tf_pos
            score_calc.fixed_tf_neg_bank = tf_neg
    elif args.train_with_img_cls_prob == 0 and not (getattr(args, "use_bayes_prompt", False) and getattr(args, "bayes_condition_on_image", True)):
        with torch.no_grad():
            text_features = _encode_text_features_mc_static(model, prompt_learner, args, device="cuda")
        score_calc.text_features = text_features

    # Avoid DataParallel overhead/extra memory on single-GPU runs.
    if torch.cuda.device_count() > 1 and len(getattr(args, "devices", [])) > 1:
        dp_calc = nn.DataParallel(score_calc)
    else:
        dp_calc = score_calc
    dp_calc.eval()
    dp_calc.cuda()

    if getattr(args, "debug_print_similarities", False):
        n_each = int(getattr(args, "debug_print_n", 5))
        _debug_print_similarity_samples(model, prompt_learner, dataloader, args, max_each=n_each)
        if getattr(args, "debug_print_only", False):
            raise SystemExit(0)

    results = {obj: {'gt_sp': [], 'pr_sp': [], 'imgs_masks': [], 'anomaly_maps': [], 'img_paths': []}
            for obj in class_details[1]}
    for items in tqdm(dataloader, desc="Processing test samples"):
        anomaly_score, anomaly_map = dp_calc(items['img'].cuda())
        anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma=args.sigma)) for i in anomaly_map.detach().cpu()], dim=0)
        mean_k = int(getattr(args, "map_mean_ksize", 0) or 0)
        if mean_k and mean_k > 1:
            # simple 2D mean filter for spatial consistency (helps AUPRO)
            am = anomaly_map.unsqueeze(1).float()
            am = F.avg_pool2d(am, kernel_size=mean_k, stride=1, padding=mean_k // 2)
            anomaly_map = am.squeeze(1)
        # Inference trick: derive image-level anomaly score from pixel-level anomaly map
        # (useful when global CLS score is unreliable after fine-tuning).
        image_score_mode = getattr(args, "image_score_mode", "cls")
        if image_score_mode != "cls":
            flat = anomaly_map.flatten(1)
            if image_score_mode == "map_max":
                anomaly_score = flat.max(dim=1).values
            elif image_score_mode == "map_topk_mean":
                k = float(getattr(args, "image_score_topk", 100))
                if 0 < k < 1:
                    kk = max(1, int(round(k * flat.shape[1])))
                else:
                    kk = int(k)
                kk = max(1, min(kk, flat.shape[1]))
                anomaly_score = flat.topk(kk, dim=1).values.mean(dim=1)
            else:
                raise ValueError(f"Unknown image_score_mode={image_score_mode!r}")

        for i in range(items['abnorm_mask'].size(0)):
            inst_cls = items['cls_id'][i].item()
            results[inst_cls]['anomaly_maps'].append(anomaly_map[i].cpu())
            results[inst_cls]['pr_sp'].append(anomaly_score[i].cpu())
            results[inst_cls]['imgs_masks'].append(items['abnorm_mask'][i].squeeze(0))
            results[inst_cls]['gt_sp'].append(items['anomaly'][i])
            results[inst_cls]['img_paths'].append(items['img_path'][i])
    
    torch.cuda.empty_cache()

    # Optional: per-class min-max normalization on image-level scores before computing metrics.
    # Note: AUROC/AP/F1-max are ranking-based (monotonic-invariant), so this typically does not
    # change them; kept for compatibility with some evaluation protocols.
    if getattr(args, "per_class_score_minmax", False):
        for obj_id, dic in results.items():
            if len(dic["pr_sp"]) == 0:
                continue
            pr = torch.stack(dic["pr_sp"]).float()
            pr_min = pr.min()
            pr_max = pr.max()
            denom = pr_max - pr_min
            if float(denom) > 0:
                pr = (pr - pr_min) / denom
            else:
                pr = torch.zeros_like(pr)
            dic["pr_sp"] = [x for x in pr.cpu()]
        
    class_names, class_ids = class_details
    if args.visualize:
        for clss, dic in results.items():
            visualizer(dic['img_paths'], dic['anomaly_maps'], dic['imgs_masks'], 518, f'{args.model_name}/{args.dataset}/{args.log_dir}/{class_names[clss]}', draw_contours=True)

    epoch_metrics = []
    for obj_id in class_ids:
        print(f'calculating metrics for {class_names[obj_id]}')
        class_metrics = compute_metrics_for_object(obj_id, results)
        class_metrics['objects'] = class_names[obj_id]
        epoch_metrics.append(class_metrics)
        
    return epoch_metrics

def generate_epoch_performance_table(epoch_metrics_dataframe, class_names):
    epoch_metrics_dataframe = pd.DataFrame(epoch_metrics_dataframe).set_index('objects')
    epoch_metrics_dataframe = epoch_metrics_dataframe.loc[class_names] # Sort
    epoch_metrics_dataframe.loc['mean'] = epoch_metrics_dataframe.mean()
    results = tabulate(epoch_metrics_dataframe, headers='keys', tablefmt="pipe", floatfmt=".03f")
    return results

def evaluate(model, items, class_details, args):
    save_path = f'{args.save_path}/{args.log_dir}/epoch_{args.epoch}/'
    logger = get_logger(save_path)
    batch_size=min(8*len(args.devices), args.batch_size) # So not to overflow the gpu memory
    print(f"process_dataset, Batch size: {batch_size}")
    # dataloader = DataLoader(items, batch_size=args.batch_size, shuffle=False)
    num_workers = int(getattr(args, "num_workers", 4))
    prefetch_factor = int(getattr(args, "prefetch_factor", 2))
    dl_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = prefetch_factor
    dataloader = DataLoader(items, **dl_kwargs)
    epoch_metrics = process_dataset(model, dataloader, class_details, args)
    # Save machine-readable metrics for later plotting/reporting.
    epoch_metrics_df = pd.DataFrame(epoch_metrics).set_index("objects")
    epoch_metrics_df = epoch_metrics_df.loc[class_details[0]]  # Sort
    epoch_metrics_df.loc["mean"] = epoch_metrics_df.mean()
    epoch_metrics_df.to_csv(os.path.join(save_path, "summary.csv"))
    epoch_metrics_df.to_json(os.path.join(save_path, "metrics.json"), orient="index")

    epoch_report = tabulate(epoch_metrics_df, headers="keys", tablefmt="pipe", floatfmt=".03f")
    
    print(args.dataset)
    logger.info("\n%s", epoch_report)
    
def test(args):        
    Crane_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth,\
                                "learnabel_text_embedding_length": args.t_n_ctx, 'others': args}
    model, _ = models.load("ViT-L/14@336px", device='cuda', design_details=Crane_parameters)
    model.visual.replace_with_EAttn(to_layer=20, type=args.attn_type)
    if args.dino_model != 'none':
        model.use_DAttn(args.dino_model)
    model = turn_gradient_off(model)
    if hasattr(model, "logit_scale"):
        try:
            ls = float(model.logit_scale.exp().detach().cpu())
            print(f"logit_scale(exp)={ls:.6f} (temp≈{1.0/ls:.6f})")
        except Exception:
            pass

    preprocess, target_transform = get_transform(args)
    test_data = Dataset(roots=args.data_path, transform=preprocess, target_transform=target_transform, \
                                dataset_name=args.dataset, kwargs=args)
    class_details = (test_data.cls_names, test_data.class_ids)

    evaluate(model, test_data, class_details, args)

if __name__ == '__main__':    
    parser = argparse.ArgumentParser("Crane", add_help=True)
    # model
    parser.add_argument("--model_name", type=str, default="trained_on_mvtec_default", help="model_name")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--visualize", type=str2bool, default=False)
    parser.add_argument("--invert_scores", type=str2bool, default=False, help="debug: invert anomaly scores/maps")
    parser.add_argument("--skip_checkpoint_load", type=str2bool, default=False, help="debug: evaluate with fresh PromptLearner init")
    parser.add_argument("--debug_print_similarities", type=str2bool, default=False, help="print cosine sims for 5 normal+5 abnormal samples")
    parser.add_argument("--debug_print_n", type=int, default=5, help="debug: number per class to print")
    parser.add_argument("--debug_print_only", type=str2bool, default=False, help="debug: exit after printing similarities")
    
    parser.add_argument("--type", type=str, default='test') 
    parser.add_argument("--devices", nargs='+', type=int, default=[0])
    parser.add_argument("--epoch", type=int, default=5) 
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="dataloader workers (reduce if RAM-limited)")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="dataloader prefetch factor (only if num_workers>0)")
    parser.add_argument(
        "--text_encode_chunk_size",
        type=int,
        default=0,
        help="chunk size for text prompt encoding (0 disables; use to avoid CUDA OOM when batch/prompts are large)",
    )
    parser.add_argument("--aug_rate", type=float, default=0.0, help="augmentation rate")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="optional explicit checkpoint path")
    parser.add_argument("--fixed_prompts", type=str2bool, default=False, help="use fixed WinCLIP-style state prompts (no learnable prompt tokens)")
    parser.add_argument("--fixed_prompt_classname", type=str, default=None, help="override class name used in fixed prompts (defaults to target_class if single class)")
    parser.add_argument("--fixed_prompts_reduce", type=str, choices=["max", "mean", "logsumexp"], default="max", help="reduction over fixed prompt bank")

    parser.add_argument("--datasets_root_dir", type=str, default=f"{DATASETS_ROOT}")
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument("--target_class", type=str, default=None, help="optional: restrict to specific class (e.g. bottle)")
    parser.add_argument("--portion", type=float, default=1) # 0.02
    parser.add_argument("--k_shot", type=int, default=0, help="number of samples per class. 0 means use all data.")

    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")    
    parser.add_argument("--train_with_img_cls_prob", type=float, default=0)
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

    # Bayes-PFL (text-side plugin)
    parser.add_argument("--use_bayes_prompt", type=str2bool, default=False)
    parser.add_argument("--bayes_num_samples", type=int, default=8)
    parser.add_argument("--bayes_flow_steps", type=int, default=4)
    parser.add_argument("--bayes_flow_type", type=str, choices=["planar", "residual"], default="planar")
    parser.add_argument("--bayes_condition_on_image", type=str2bool, default=True)
    parser.add_argument("--bayes_init_logstd", type=float, default=math.log(0.02))
    parser.add_argument("--bayes_use_residual", type=str2bool, default=True, help="residual-gate Bayes prompt updates")
    parser.add_argument("--bayes_residual_alpha_init", type=float, default=0.01, help="init alpha for residual-gated Bayes prompts")

    # Feature refinement (attention-ish, minimal-risk)
    parser.add_argument("--frm_type", type=str, choices=["scalar", "linear"], default="scalar")
    parser.add_argument("--frm_alpha_init", type=float, default=0.0)

    parser.add_argument("--dino_model", type=str, choices=['none', 'dinov2', 'dino', 'sam'], default='dinov2')
    parser.add_argument("--use_scorebase_pooling", type=str2bool, default=True)

    parser.add_argument("--image_size", type=int, default=518, help="input image size")
    parser.add_argument("--features_list", type=int, nargs="+", default=[24], help="layer features used")
    parser.add_argument("--sigma", type=int, default=4, help="zero shot")
    parser.add_argument("--map_mean_ksize", type=int, default=0, help="optional mean filter kernel size on anomaly map (0 disables)")
    parser.add_argument("--soft_mean", type=str2bool, default=False) 

    # Multi-scale fusion (VFM + CLIP patch tokens)
    parser.add_argument("--vfm_fusion", type=str2bool, default=False, help="fuse VFM (e.g., DINOv2) intermediate features into CLIP patch tokens")
    parser.add_argument("--vfm_num_layers", type=int, default=4, help="number of VFM intermediate layers to average for fusion")
    parser.add_argument("--vfm_fusion_weight", type=float, default=1.0, help="weight of VFM tokens in fusion")
    parser.add_argument("--vfm_fusion_mode", type=str, choices=["concat_mean", "add"], default="concat_mean", help="fusion mode for CLIP+VFM tokens")
    parser.add_argument("--patch_graph_smooth", type=str2bool, default=False, help="apply neighbor smoothing on patch tokens")
    parser.add_argument("--patch_graph_lambda", type=float, default=0.1, help="smoothing strength for patch_graph_smooth")
    parser.add_argument(
        "--per_class_score_minmax",
        type=str2bool,
        default=False,
        help="apply per-class min-max normalization to image-level scores before computing metrics",
    )
    parser.add_argument(
        "--image_score_mode",
        type=str,
        choices=["cls", "map_max", "map_topk_mean"],
        default="cls",
        help="image anomaly score source: cls logits vs pixel anomaly map (max/topk-mean)",
    )
    parser.add_argument(
        "--image_score_topk",
        type=float,
        default=100,
        help="top-k for map_topk_mean; if in (0,1) treated as fraction of pixels",
    )

    parser.add_argument("--attn_type", type=str, choices=["vv", "kk", "qq", "qq+kk", "qq+kk+vv", "(q+k+v)^2"], default="qq+kk+vv")
    parser.add_argument("--both_eattn_dattn", type=str2bool, default=True)

    parser.add_argument("--why", type=str, help="Explanation about this experiment and how it is different other than parameter values")
    args = parser.parse_args()
    
    if 'CUDA_VISIBLE_DEVICES' not in os.environ: # Forcing all the tensors to be on the specified device(s)
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.devices)) if len(args.devices) > 1 else str(args.devices[0])
        command = [sys.executable,  __file__, ] + sys.argv[1:] 
        process = subprocess.Popen(command, env=os.environ)
        process.wait()
        
    else:
        setup_seed(args.seed)
        args.log_dir = make_human_readable_name(args)        
        
        args.data_path = [f"{args.datasets_root_dir}/{args.dataset}/"]
        if args.checkpoint_path is None:
            args.checkpoint_path = f'./checkpoints/{args.model_name}/epoch_{args.epoch}.pth'
        args.save_path = f'./results/{args.model_name}/test_on_{args.dataset}/'

        print(f"Testing on dataset from: {args.data_path}") 
        print(f"Results will be saved to: {colored(args.save_path+args.log_dir, 'green')}")

        save_args_to_file(args, sys.argv[1:], log_dir=args.log_dir)

        test(args)
        
