#!/usr/bin/env bash
set -euo pipefail

# Runs the required ablations:
# - Baseline (Modified WinCLIP): Crane w/ E-Attn, no DINO, no Bayes, no FRM
# - +DINO: enable DINOv2
# - +Bayes: enable Bayesian prompt flow plugin (text-side)
# - +Attention: enable feature_refinement_module (minimal-risk)
#
# Usage:
#   bash main/experiments/run_ablation.sh <device_id>

DEVICE="${1:?device_id required}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

COMMON_TEST_ARGS=(
  --train_with_img_cls_prob 0
  --train_with_img_cls_type none
)

echo "[1/4] Baseline (Modified WinCLIP)"
bash "${ROOT_DIR}/experiments/run_cross_domain.sh" "baseline" "${DEVICE}" \
  --train --dino_model none \
  --test  --dino_model none "${COMMON_TEST_ARGS[@]}"

echo "[2/4] +DINO"
bash "${ROOT_DIR}/experiments/run_cross_domain.sh" "plus_dino" "${DEVICE}" \
  --train --dino_model dinov2 \
  --test  --dino_model dinov2 "${COMMON_TEST_ARGS[@]}"

echo "[3/4] +Bayes"
bash "${ROOT_DIR}/experiments/run_cross_domain.sh" "plus_bayes" "${DEVICE}" \
  --train --dino_model none --use_bayes_prompt True --bayes_condition_on_image True --train_with_img_cls_prob 0 --train_with_img_cls_type none \
  --test  --dino_model none --use_bayes_prompt True --bayes_condition_on_image True "${COMMON_TEST_ARGS[@]}"

echo "[4/4] +Attention (feature_refinement_module)"
bash "${ROOT_DIR}/experiments/run_cross_domain.sh" "plus_attention" "${DEVICE}" \
  --train --dino_model none --use_feature_refinement_module True --frm_type scalar \
  --test  --dino_model none --frm_type scalar "${COMMON_TEST_ARGS[@]}"
