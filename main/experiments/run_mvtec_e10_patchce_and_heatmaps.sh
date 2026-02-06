#!/usr/bin/env bash
set -euo pipefail

# Runs:
# 1) Train on MVTec(train/good only) for EPOCHS (default 10)
# 2) In-domain eval: trained_on_mvtec_<TAG> -> mvtec
# 3) Heatmap visualization on 3 classes: bottle,cable,metal_nut
#
# Usage:
#   bash main/experiments/run_mvtec_e10_patchce_and_heatmaps.sh <tag> <device_id>
#
# Env:
#   EPOCHS=10
#   CRANE_DATASETS_ROOT=/home/jovyan/data

TAG="${1:?tag required}"
DEVICE="${2:?device_id required}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRANE_DIR="${ROOT_DIR}/Crane-main"

EPOCHS="${EPOCHS:-10}"
export CRANE_DATASETS_ROOT="${CRANE_DATASETS_ROOT:-/home/jovyan/data}"
TRAIN_BATCH="${TRAIN_BATCH:-64}"
VRAM_RESERVE_FRAC="${VRAM_RESERVE_FRAC:-0}"

export STOP_AFTER_MVTEC_EVAL=1
export EPOCHS

# Train + in-domain eval
bash "${ROOT_DIR}/experiments/run_cross_domain.sh" "${TAG}" "${DEVICE}" \
  --train \
    --use_bayes_prompt True \
    --bayes_align_official True \
    --bayes_num_samples_train 1 \
    --learning_rate 0.0001 \
    --bayes_img_ce_weight 0.2 \
    --bayes_patch_ce_weight 1.0 \
    --bayes_pfl_weight 1.0 \
  --dino_model dinov2 \
  --batch_size "${TRAIN_BATCH}" \
    --vram_reserve_frac "${VRAM_RESERVE_FRAC}" \
    --num_workers 4 \
    --prefetch_factor 2 \
  --test \
    --use_bayes_prompt True \
    --bayes_condition_on_image True \
    --bayes_num_samples 8 \
    --dino_model dinov2 \
    --batch_size 32 \
    --num_workers 4 \
    --prefetch_factor 2

# Heatmap spot-check
cd "${CRANE_DIR}"
python3 test.py \
  --dataset mvtec \
  --target_class bottle,cable,metal_nut \
  --model_name "trained_on_mvtec_${TAG}" \
  --epoch "${EPOCHS}" \
  --devices "${DEVICE}" \
  --dino_model dinov2 \
  --use_bayes_prompt True \
  --bayes_condition_on_image True \
  --bayes_num_samples 8 \
  --train_with_img_cls_prob 0 \
  --batch_size 32 \
  --num_workers 4 \
  --prefetch_factor 2 \
  --visualize True
