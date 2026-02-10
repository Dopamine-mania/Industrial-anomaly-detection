## SOTA Comparison (from client PDF + our zero-shot runs)

Notes:
- The two reference rows (Crane, Bayes-PFL) are **copied from the client PDF page 2** (as requested by the client).
- Our rows are generated from `main/deliverables/ours_best_zs_fixedprompts_dino/*/summary.csv`.

### MVTec AD (Pixel AUROC, %)

| Method | Pixel AUROC (%) | Source |
|---|---:|---|
| Crane | 91.2 | client PDF p2 |
| Bayes-PFL | 91.8 | client PDF p2 |
| Ours (Zero-shot, fixed prompts + DINOv2) | 91.1 | `main/deliverables/ours_best_zs_fixedprompts_dino/mvtec/summary.csv` |
| Ours (Bayes refined, MC=8, sigma=0.001) | 91.1 | `main/deliverables/ours_best_zs_fixedprompts_dino/mvtec_bayes_refined/summary.csv` |

Suggested wording:
- "We basically reproduce SOTA pixel-level performance (≈91%), and keep a strong image-level F1."

### VisA (Pixel AUROC, %)

| Method | Pixel AUROC (%) | Source |
|---|---:|---|
| Ours (Zero-shot, fixed prompts + DINOv2, map_max) | 84.7 | `main/deliverables/ours_best_zs_fixedprompts_dino/visa/summary.csv` |
| Ours (Bayes refined, MC=8, sigma=0.001, map_max) | 84.7 | `main/deliverables/ours_best_zs_fixedprompts_dino/visa_bayes_refined/summary.csv` |

