#!/usr/bin/env bash
set -euo pipefail

# Tuned "pixel-first" run:
# - resume from an existing checkpoint (optional)
# - heavy patch-level CE, light/zero image-level CE
# - low LR + small weight decay
# - train MVTec (train/good only), then in-domain eval, then visualize 3 classes
#
# Usage:
#   bash main/experiments/run_mvtec_e30_tuned_and_heatmaps.sh <tag> <device_id>
#
# Env (defaults):
#   CRANE_DATASETS_ROOT=/home/jovyan/data
#   EPOCHS=30
#   TRAIN_BATCH=80
#   LR=5e-5
#   WEIGHT_DECAY=1e-6
#   IMG_CE_W=0
#   PATCH_CE_W=1.0
#   PFL_W=1.0
#   RESUME_PATH= (optional, e.g. /.../epoch_10.pth)

TAG="${1:?tag required}"
DEVICE="${2:?device_id required}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRANE_DIR="${ROOT_DIR}/Crane-main"

export CRANE_DATASETS_ROOT="${CRANE_DATASETS_ROOT:-/home/jovyan/data}"
export EPOCHS="${EPOCHS:-30}"

TRAIN_BATCH="${TRAIN_BATCH:-80}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
LR="${LR:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}"
IMG_CE_W="${IMG_CE_W:-0}"
PATCH_CE_W="${PATCH_CE_W:-1.0}"
PFL_W="${PFL_W:-1.0}"
RESUME_PATH="${RESUME_PATH:-}"

export STOP_AFTER_MVTEC_EVAL=1

TRAIN_EXTRA=(
  --use_bayes_prompt True
  --bayes_align_official True
  --bayes_num_samples_train 1
  --learning_rate "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --bayes_img_ce_weight "${IMG_CE_W}"
  --bayes_patch_ce_weight "${PATCH_CE_W}"
  --bayes_pfl_weight "${PFL_W}"
  --dino_model dinov2
  --batch_size "${TRAIN_BATCH}"
  --num_workers "${NUM_WORKERS}"
  --prefetch_factor "${PREFETCH_FACTOR}"
  --vram_reserve_frac 0
)
if [[ -n "${RESUME_PATH}" ]]; then
  TRAIN_EXTRA+=(--resume_path "${RESUME_PATH}")
fi

TEST_EXTRA=(
  --use_bayes_prompt True
  --bayes_condition_on_image True
  --bayes_num_samples 8
  --dino_model dinov2
  --batch_size 32
  --num_workers "${NUM_WORKERS}"
  --prefetch_factor "${PREFETCH_FACTOR}"
)

bash "${ROOT_DIR}/experiments/run_cross_domain.sh" "${TAG}" "${DEVICE}" \
  --train "${TRAIN_EXTRA[@]}" \
  --test "${TEST_EXTRA[@]}"

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
  --num_workers "${NUM_WORKERS}" \
  --prefetch_factor "${PREFETCH_FACTOR}" \
  --visualize True
