# Industrial-anomaly-detection — Progress (2026-02-09)

## Scope / Constraints
- Work directory: `main/` only (do not modify `客户需求/`).
- Datasets: **MVTec AD** and **VisA** only.
- Training data rule: if training is used, **only** `train/good` (never touch `test/*` for training).

## What is done
### Code changes (Crane-main base)
- `main/Crane-main/utils/metrics.py`: makes `pixel_aupro` robust by removing hard dependency on `skimage` and falling back to an alternate implementation when needed.
- `main/Crane-main/test.py`:
  - Adds **Bayes-style MC sampling** for *fixed prompts* at inference (`--fixed_prompts_bayes ...`).
  - Adds `--use_feature_refinement_module` switch for the “+Attention” ablation (inference-time toggle).

### Locked “Ours (Best)” results (Zero-shot + fixed prompts + DINOv2)
Backups live in `main/deliverables/ours_best_zs_fixedprompts_dino/`.

- MVTec (mean): Pixel AUROC **0.9109**, Pixel AUPRO **0.8773**, Image AUROC **0.8650**
  - Source: `main/deliverables/ours_best_zs_fixedprompts_dino/mvtec/summary.csv`
- VisA (mean, `image_score_mode=map_max`): Pixel AUROC **0.847**, Image AUROC **0.811**
  - Source: `main/deliverables/ours_best_zs_fixedprompts_dino/visa/summary.csv`

### “Ours (Bayes refined)” (inference-only, MC=8, sigma=0.001)
- MVTec: essentially identical to Best
  - Source: `main/deliverables/ours_best_zs_fixedprompts_dino/mvtec_bayes_refined/summary.csv`
- VisA: essentially identical to Best
  - Source: `main/deliverables/ours_best_zs_fixedprompts_dino/visa_bayes_refined/summary.csv`

### Aggregation artifacts for PM
- Final table: `main/analysis/final_table.md` and `main/analysis/final_table.csv`
- Auto-collected run index: `main/analysis/summary.csv`
- Ablation plots (current): `main/analysis/ablation_pixel_auroc_mvtec.png`, `main/analysis/ablation_image_auroc_mvtec.png`

## In progress
- Heatmap visualizations for a few representative classes (Hazelnut pending).

## Newly completed (since last update)
- “+Attention” ablation (FRM scalar) on MVTec (mean metrics are essentially unchanged vs +Bayes).
- Heatmap samples exported for report:
  - `main/deliverables/heatmaps/mvtec/bottle/`
  - `main/deliverables/heatmaps/mvtec/cable/`

## Next actions
- Finish “+Attention” eval and regenerate ablation plots with 4 bars.
- Generate 6–12 representative `vis_img` pairs (`*_orig.png`, `*_pred.png`) for report.
- Optional (if required by PM): run **trained-on-MVTec** and **trained-on-VisA** checkpoints for “trained_on_A → test_on_B” cross-domain tables.
