#!/usr/bin/env bash
set -euo pipefail

# Robust MVTec resume runner:
# - keeps trying until epoch_30.pth exists
# - resumes from latest epoch_*.pth (or a provided RESUME_PATH)
# - writes a single rolling log file
#
# Usage:
#   bash main/experiments/watch_mvtec_resume_to_e30.sh <tag> <device_id>
#
# Env (defaults):
#   CRANE_DATASETS_ROOT=/home/jovyan/data
#   EPOCHS=30
#   TRAIN_BATCH=64
#   NUM_WORKERS=2
#   PREFETCH_FACTOR=2
#   LR=5e-5
#   WEIGHT_DECAY=1e-6
#   IMG_CE_W=0
#   PATCH_CE_W=1.0
#   PFL_W=1.0
#   RESUME_PATH= (optional explicit checkpoint)
#   SLEEP_ON_FAIL=60
#   NO_ALBUMENTATIONS_UPDATE=1

TAG="${1:?tag required}"
DEVICE="${2:?device_id required}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRANE_DIR="${ROOT_DIR}/Crane-main"

export CRANE_DATASETS_ROOT="${CRANE_DATASETS_ROOT:-/home/jovyan/data}"
export EPOCHS="${EPOCHS:-30}"

TRAIN_BATCH="${TRAIN_BATCH:-64}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
LR="${LR:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}"
IMG_CE_W="${IMG_CE_W:-0}"
PATCH_CE_W="${PATCH_CE_W:-1.0}"
PFL_W="${PFL_W:-1.0}"
RESUME_PATH="${RESUME_PATH:-}"
SLEEP_ON_FAIL="${SLEEP_ON_FAIL:-60}"

export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export PYTHONUNBUFFERED=1

RUNS_DIR="${ROOT_DIR}/runs"
mkdir -p "${RUNS_DIR}"
LOG="${RUNS_DIR}/${TAG}_watch_$(date +%Y%m%d_%H%M%S).log"
echo "${LOG}" > "${RUNS_DIR}/${TAG}.logpath"

SAVE_DIR="${CRANE_DIR}/checkpoints/trained_on_mvtec_${TAG}"
mkdir -p "${SAVE_DIR}"

_latest_ckpt() {
  if [[ -n "${RESUME_PATH}" ]]; then
    echo "${RESUME_PATH}"
    return 0
  fi
  ls -t "${SAVE_DIR}"/epoch_*.pth 2>/dev/null | head -n 1 || true
}

_cuda_ready() {
  python3 - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1)
PY
}

echo "[$(date -u '+%F %T UTC')] watch start: tag=${TAG}, device=${DEVICE}" | tee -a "${LOG}"

while true; do
  if [[ -f "${SAVE_DIR}/epoch_${EPOCHS}.pth" ]]; then
    echo "[$(date -u '+%F %T UTC')] epoch_${EPOCHS}.pth exists; done." | tee -a "${LOG}"
    break
  fi

  if ! _cuda_ready; then
    echo "[$(date -u '+%F %T UTC')] CUDA not available; sleep ${SLEEP_ON_FAIL}s" | tee -a "${LOG}"
    sleep "${SLEEP_ON_FAIL}"
    continue
  fi

  CKPT="$(_latest_ckpt)"
  if [[ -n "${CKPT}" ]]; then
    echo "[$(date -u '+%F %T UTC')] resume from: ${CKPT}" | tee -a "${LOG}"
  else
    echo "[$(date -u '+%F %T UTC')] no checkpoint found; start from scratch" | tee -a "${LOG}"
  fi

  export CUDA_VISIBLE_DEVICES="${DEVICE}"
  cd "${CRANE_DIR}"

  set +e
  python3 -u train.py \
    --dataset mvtec \
    --model_name "${TAG}" \
    --device 0 \
    --epoch "${EPOCHS}" \
    --save_freq 1 \
    --train_good_only True \
    --datasets_root_dir "${CRANE_DATASETS_ROOT}" \
    --use_bayes_prompt True \
    --bayes_align_official True \
    --bayes_num_samples_train 1 \
    --learning_rate "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --bayes_img_ce_weight "${IMG_CE_W}" \
    --bayes_patch_ce_weight "${PATCH_CE_W}" \
    --bayes_pfl_weight "${PFL_W}" \
    --dino_model dinov2 \
    --batch_size "${TRAIN_BATCH}" \
    --num_workers "${NUM_WORKERS}" \
    --prefetch_factor "${PREFETCH_FACTOR}" \
    --vram_reserve_frac 0 \
    ${CKPT:+--resume_path "${CKPT}"} \
    2>&1 | tee -a "${LOG}"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ $rc -eq 0 ]]; then
    echo "[$(date -u '+%F %T UTC')] train.py exited 0" | tee -a "${LOG}"
  else
    echo "[$(date -u '+%F %T UTC')] train.py exited ${rc}; sleep ${SLEEP_ON_FAIL}s then retry" | tee -a "${LOG}"
    sleep "${SLEEP_ON_FAIL}"
  fi
done

echo "[$(date -u '+%F %T UTC')] running in-domain eval + heatmaps" | tee -a "${LOG}"
export CUDA_VISIBLE_DEVICES="${DEVICE}"
cd "${CRANE_DIR}"
python3 -u test.py \
  --dataset mvtec \
  --model_name "trained_on_mvtec_${TAG}" \
  --epoch "${EPOCHS}" \
  --devices 0 \
  --datasets_root_dir "${CRANE_DATASETS_ROOT}" \
  --dino_model dinov2 \
  --use_bayes_prompt True \
  --bayes_condition_on_image True \
  --bayes_num_samples 8 \
  --train_with_img_cls_prob 0 \
  --batch_size 32 \
  --num_workers "${NUM_WORKERS}" \
  --prefetch_factor "${PREFETCH_FACTOR}" \
  --visualize False \
  2>&1 | tee -a "${LOG}"

python3 -u test.py \
  --dataset mvtec \
  --target_class bottle,cable,metal_nut \
  --model_name "trained_on_mvtec_${TAG}" \
  --epoch "${EPOCHS}" \
  --devices 0 \
  --datasets_root_dir "${CRANE_DATASETS_ROOT}" \
  --dino_model dinov2 \
  --use_bayes_prompt True \
  --bayes_condition_on_image True \
  --bayes_num_samples 8 \
  --train_with_img_cls_prob 0 \
  --batch_size 32 \
  --num_workers "${NUM_WORKERS}" \
  --prefetch_factor "${PREFETCH_FACTOR}" \
  --visualize True \
  2>&1 | tee -a "${LOG}"

echo "[$(date -u '+%F %T UTC')] done" | tee -a "${LOG}"

