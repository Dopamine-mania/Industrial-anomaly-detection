# Crane + Bayes-PFL (plugin) + ResCLIP — Customer Guide

This delivery is based on the **Crane** codebase (DINOv2 + D-Attn/E-Attn pipeline), with a **Bayes-PFL-inspired** prompt sampling plugin and an explicit **ResCLIP** residual fusion module for dense inference.

## Package layout
- `code/` — runnable code (Crane-main + plugins)
- `checkpoints/` — `best_model_mvtec.pth` (note: best setting is **zero-shot / training-free**; this file is a provenance manifest)
- `results/` — final tables and plots
- `heatmaps/` — sample heatmap visualizations (orig + pred overlay)

## Environment
Recommended: Python 3.10+ with CUDA.

Key pip deps (minimal):
- `torch`, `torchvision`
- `opencv-python`
- `scipy`
- `torchmetrics`
- `tabulate`
- `matplotlib`

## Data preparation
Crane expects each dataset root to contain a `meta.json`.

Typical dataset layout (standard):
- MVTec AD: `train/good`, `test/<defect_name>`, `ground_truth/<defect_name>`
- VisA: standard `train`/`test` with masks

Generate `meta.json` automatically:
```bash
cd code/Crane-main
python utils/generate_meta.py --dataset_root /path/to/mvtec_ad --out_meta /path/to/mvtec_ad/meta.json
python utils/generate_meta.py --dataset_root /path/to/visa --out_meta /path/to/visa/meta.json
```

## Reproduce the main numbers (zero-shot)
### MVTec AD (Best / Ours)
```bash
cd code/Crane-main
CUDA_VISIBLE_DEVICES=0 python test.py \
  --dataset mvtec --datasets_root_dir /path/to/datasets_root \
  --model_name zs_fixedprompts_mvtec_best \
  --skip_checkpoint_load True \
  --dino_model dinov2 \
  --fixed_prompts True --fixed_prompts_reduce mean \
  --sigma 4 --image_size 518
```

### VisA (Best / Ours, image_score_mode=map_max)
```bash
cd code/Crane-main
CUDA_VISIBLE_DEVICES=0 python test.py \
  --dataset visa --datasets_root_dir /path/to/datasets_root \
  --model_name zs_fixedprompts_visa_best \
  --skip_checkpoint_load True \
  --dino_model dinov2 \
  --fixed_prompts True --fixed_prompts_reduce mean \
  --image_score_mode map_max \
  --sigma 4 --image_size 518
```

## Enable Bayes (MC sampling) + ResCLIP residual fusion
Bayes sampling is done at inference by perturbing **fixed-prompt token embeddings** and averaging (`num_samples=8`).

ResCLIP explicitly fuses the **Crane branch** (deterministic prompts) and the **Bayes branch** (MC prompts):
`Final = (1 - alpha) * Crane + alpha * Bayes`.

```bash
cd code/Crane-main
CUDA_VISIBLE_DEVICES=0 python test.py \
  --dataset mvtec --datasets_root_dir /path/to/datasets_root \
  --model_name ours_bayes_resclip \
  --skip_checkpoint_load True \
  --dino_model dinov2 \
  --fixed_prompts True --fixed_prompts_reduce mean \
  --fixed_prompts_bayes True --fixed_prompts_bayes_num_samples 8 \
  --fixed_prompts_bayes_init_logstd -6.907755278982137 \
  --use_resclip True --resclip_alpha 0.5 \
  --sigma 4 --image_size 518
```

## Generate heatmaps (sample)
```bash
cd code/Crane-main
CUDA_VISIBLE_DEVICES=0 python test.py \
  --dataset mvtec --datasets_root_dir /path/to/datasets_root \
  --model_name demo_heatmap_bottle \
  --skip_checkpoint_load True \
  --dino_model dinov2 \
  --fixed_prompts True --fixed_prompts_reduce mean \
  --target_class bottle --portion 0.05 \
  --visualize True
```
Output goes to `code/Crane-main/vis_img/`.

## Final tables / plots
- `results/final_table.csv` — final metrics table
- `results/sota_comparison.png` — SOTA comparison bar chart (Pixel AUROC)
- `results/variance_analysis.png` — Bayes MC sampling variance histogram

