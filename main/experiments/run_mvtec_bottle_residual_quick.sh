#!/usr/bin/env bash
set -euo pipefail

# Quick sanity run for the "residual-guided" Bayes prompt training on a single class (Bottle).
# Trains on train/good only and evaluates in-domain on Bottle.
#
# Usage:
#   bash main/experiments/run_mvtec_bottle_residual_quick.sh <tag> <device_id>
#
# Env (defaults):
#   CRANE_DATASETS_ROOT=/home/jovyan/data
#   EPOCHS=5
#   LR=1e-5
#   TRAIN_BATCH=64
#   N_CTX=12
#   CTX_INIT=clip
#   CTX_INIT_PHRASE="a photo of a"
#   TARGET_CLASS=bottle
#   BAYES_KL_WEIGHT=0.01
#   TEXT_ENCODE_CHUNK=256

TAG="${1:?tag required}"
DEVICE="${2:?device_id required}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRANE_DIR="${ROOT_DIR}/Crane-main"

export CRANE_DATASETS_ROOT="${CRANE_DATASETS_ROOT:-/home/jovyan/data}"
export EPOCHS="${EPOCHS:-5}"
export LR="${LR:-1e-5}"
export TRAIN_BATCH="${TRAIN_BATCH:-64}"
export N_CTX="${N_CTX:-12}"
export CTX_INIT="${CTX_INIT:-clip}"
export CTX_INIT_PHRASE="${CTX_INIT_PHRASE:-a photo of a}"
export TARGET_CLASS="${TARGET_CLASS:-bottle}"
export BAYES_KL_WEIGHT="${BAYES_KL_WEIGHT:-0.01}"
export TEXT_ENCODE_CHUNK="${TEXT_ENCODE_CHUNK:-256}"

export CUDA_VISIBLE_DEVICES="${DEVICE}"
INTERNAL_DEVICE_ID=0

cd "${CRANE_DIR}"

python3 -u train.py \
  --dataset mvtec \
  --target_class "${TARGET_CLASS}" \
  --model_name "${TAG}" \
  --device "${INTERNAL_DEVICE_ID}" \
  --epoch "${EPOCHS}" \
  --save_freq 1 \
  --train_good_only True \
  --datasets_root_dir "${CRANE_DATASETS_ROOT}" \
  --dino_model dinov2 \
  --n_ctx "${N_CTX}" \
  --batch_size "${TRAIN_BATCH}" \
  --num_workers 0 \
  --text_encode_chunk_size "${TEXT_ENCODE_CHUNK}" \
  --learning_rate "${LR}" \
  --weight_decay 1e-6 \
  --prompt_train_mode bayes_only \
  --ctx_init "${CTX_INIT}" \
  --ctx_init_phrase "${CTX_INIT_PHRASE}" \
  --use_bayes_prompt True \
  --bayes_align_official True \
  --bayes_use_residual True \
  --bayes_residual_alpha_init 0.01 \
  --bayes_num_samples_train 1 \
  --bayes_kl_weight "${BAYES_KL_WEIGHT}" \
  --bayes_img_ce_weight 0 \
  --bayes_patch_loss ce \
  --bayes_patch_ce_weight 1.0 \
  --bayes_pfl_weight 1.0 \
  --synthetic_anomaly_prob 0.5 \
  --synthetic_anomaly_mode cutpaste \
  --synthetic_anomaly_area_min 0.02 \
  --synthetic_anomaly_area_max 0.15

python3 -u test.py \
  --dataset mvtec \
  --target_class "${TARGET_CLASS}" \
  --model_name "trained_on_mvtec_${TAG}" \
  --epoch "${EPOCHS}" \
  --devices "${INTERNAL_DEVICE_ID}" \
  --datasets_root_dir "${CRANE_DATASETS_ROOT}" \
  --dino_model dinov2 \
  --n_ctx "${N_CTX}" \
  --use_bayes_prompt True \
  --bayes_condition_on_image True \
  --bayes_num_samples 16 \
  --sigma 4 \
  --image_score_mode map_max \
  --batch_size 1 \
  --num_workers 0 \
  --text_encode_chunk_size "${TEXT_ENCODE_CHUNK}"
