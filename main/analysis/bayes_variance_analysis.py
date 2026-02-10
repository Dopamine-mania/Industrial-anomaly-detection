import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _reduce_prompt_scores(scores: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "max":
        return scores.max(dim=-1).values
    if mode == "mean":
        return scores.mean(dim=-1)
    if mode == "logsumexp":
        return torch.logsumexp(scores, dim=-1)
    raise ValueError(f"Unknown reduce={mode!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, choices=["mvtec", "visa"], default="mvtec")
    ap.add_argument("--datasets_root_dir", type=str, default="/home/jovyan/data")
    ap.add_argument("--target_class", type=str, default="bottle")
    ap.add_argument("--n_images", type=int, default=64)
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--init_logstd", type=float, default=math.log(0.001))
    ap.add_argument("--reduce", type=str, choices=["max", "mean", "logsumexp"], default="mean")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out_dir", type=Path, default=Path("main/analysis/bayes_variance"))
    args = ap.parse_args()

    # Make Crane-main importable when running from repo root.
    import sys
    crane_root = Path(__file__).resolve().parents[1] / "Crane-main"
    sys.path.insert(0, str(crane_root))

    # Local imports from Crane-main
    import models
    from dataset.dataset import Dataset
    from utils.transform import get_transform
    from models.prompt_ensemble import tokenize as clip_tokenize
    from models import prompt_ensemble as _pe
    from models.state_prompts import NORMAL_STATE_TEMPLATES, ABNORMAL_STATE_TEMPLATES

    device = "cuda" if torch.cuda.is_available() else "cpu"

    crane_args = argparse.Namespace(
        n_ctx=12,
        depth=9,
        t_n_ctx=4,
        attn_type="qq+kk+vv",
        both_eattn_dattn=True,
        dino_model="dinov2",
        image_size=518,
        features_list=[24],
        type="test",
        aug_rate=0.0,
        portion=1.0,
        k_shot=0,
        train_with_img_cls_prob=0.0,
        train_with_img_cls_type="none",
        fixed_prompts=True,
        fixed_prompts_reduce=args.reduce,
        fixed_prompt_classname=args.target_class,
        ctx_init="random",
        ctx_init_phrase="a photo of a",
        datasets_root_dir=args.datasets_root_dir,
        dataset=args.dataset,
        target_class=args.target_class,
    )

    design_details = {"Prompt_length": crane_args.n_ctx, "learnabel_text_embedding_depth": crane_args.depth, "learnabel_text_embedding_length": crane_args.t_n_ctx, "others": crane_args}
    model, _ = models.load("ViT-L/14@336px", device=device, design_details=design_details)
    if hasattr(model.visual, "replace_with_EAttn"):
        model.visual.replace_with_EAttn(to_layer=20, type=crane_args.attn_type)
    if getattr(crane_args, "both_eattn_dattn", True) and hasattr(model.visual, "replace_with_DAttn"):
        model.visual.replace_with_DAttn(to_layer=20, type=crane_args.attn_type)
    model.eval()

    preprocess, target_transform = get_transform(crane_args)
    ds = Dataset(
        roots=[f"{args.datasets_root_dir}/{args.dataset}/"],
        transform=preprocess,
        target_transform=target_transform,
        dataset_name=args.dataset,
        kwargs=crane_args,
    )
    # Restrict to class via Dataset(target_class=...) already; keep only a subset for speed.
    n = min(int(args.n_images), len(ds))
    subset = torch.utils.data.Subset(ds, list(range(n)))
    dl = DataLoader(subset, batch_size=int(args.batch_size), shuffle=False, num_workers=2, pin_memory=True)

    # Prepare prompt tokens
    prompts_pos = [t.format(args.target_class) for t in NORMAL_STATE_TEMPLATES]
    prompts_neg = [t.format(args.target_class) for t in ABNORMAL_STATE_TEMPLATES]
    tokens_pos = clip_tokenize(prompts_pos).to(device=device)
    tokens_neg = clip_tokenize(prompts_neg).to(device=device)

    cast_dtype = model.transformer.get_cast_dtype()
    base_emb_pos = model.token_embedding(tokens_pos).type(cast_dtype)
    base_emb_neg = model.token_embedding(tokens_neg).type(cast_dtype)

    sot_id = int(_pe._tokenizer.encoder["<|startoftext|>"])
    eot_id = int(_pe._tokenizer.encoder["<|endoftext|>"])
    m_pos = ((tokens_pos != 0) & (tokens_pos != sot_id) & (tokens_pos != eot_id)).unsqueeze(-1).to(dtype=base_emb_pos.dtype)
    m_neg = ((tokens_neg != 0) & (tokens_neg != sot_id) & (tokens_neg != eot_id)).unsqueeze(-1).to(dtype=base_emb_neg.dtype)

    sigma = float(math.exp(float(args.init_logstd)))

    # Collect per-image anomaly probs across MC samples
    all_scores = []
    all_paths = []
    all_gt = []

    with torch.no_grad():
        for batch in dl:
            imgs = batch["img"].to(device=device, non_blocking=True)
            gts = batch["anomaly"].cpu().numpy().tolist()
            paths = batch["img_path"]

            img_feat, _ = model.encode_image(imgs, crane_args.features_list, self_cor_attn_layers=20)
            img_feat = F.normalize(img_feat, dim=-1)

            sample_scores = []
            for _ in range(int(args.num_samples)):
                emb_pos = base_emb_pos + sigma * torch.randn_like(base_emb_pos) * m_pos
                emb_neg = base_emb_neg + sigma * torch.randn_like(base_emb_neg) * m_neg
                tf_pos = model.encode_text_learn(emb_pos, tokens_pos, deep_compound_prompts_text=[]).float()
                tf_neg = model.encode_text_learn(emb_neg, tokens_neg, deep_compound_prompts_text=[]).float()
                tf_pos = F.normalize(tf_pos, dim=-1)
                tf_neg = F.normalize(tf_neg, dim=-1)

                pos = (img_feat @ tf_pos.t()) / 0.07
                neg = (img_feat @ tf_neg.t()) / 0.07
                pos_r = _reduce_prompt_scores(pos, args.reduce)
                neg_r = _reduce_prompt_scores(neg, args.reduce)
                logits = torch.stack([pos_r, neg_r], dim=-1)
                prob = logits.softmax(dim=-1)[:, 1]
                sample_scores.append(prob.detach().cpu())

            sample_scores = torch.stack(sample_scores, dim=0)  # (S,B)
            all_scores.append(sample_scores)
            all_paths.extend(list(paths))
            all_gt.extend(gts)

    scores = torch.cat(all_scores, dim=1)  # (S,N)
    mean = scores.mean(dim=0)
    std = scores.std(dim=0, unbiased=False)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / f"{args.dataset}_{args.target_class}_mc{args.num_samples}_sigma{sigma:.4g}.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["img_path", "gt_anomaly", "score_mean", "score_std"])
        for p, y, m, s in zip(all_paths, all_gt, mean.tolist(), std.tolist()):
            w.writerow([p, int(y), float(m), float(s)])

    # Simple plot (optional)
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 3))
        plt.hist(std.numpy(), bins=30)
        plt.title(f"MC score std (S={args.num_samples}, sigma={sigma:.4g})")
        plt.xlabel("std")
        plt.ylabel("count")
        plt.tight_layout()
        out_png = args.out_dir / f"{args.dataset}_{args.target_class}_mc_std_hist.png"
        plt.savefig(out_png, dpi=200)
        print(f"Wrote {out_png}")
    except Exception:
        pass

    print(f"Wrote {out_csv}")
    print(f"mean(std)={float(std.mean()):.6f}  max(std)={float(std.max()):.6f}")


if __name__ == "__main__":
    main()
