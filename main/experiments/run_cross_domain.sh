#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash main/experiments/run_cross_domain.sh <tag> <device_id> \
#     --train [extra_train_args...] --test [extra_test_args...]
#
# This trains two checkpoints:
# - trained_on_mvtec_<tag>
# - trained_on_visa_<tag>
# and (optionally) runs in-domain sanity checks:
# - trained_on_mvtec_<tag> -> mvtec
# - trained_on_visa_<tag>  -> visa
# and evaluates cross-domain:
# - mvtec -> visa
# - visa  -> mvtec
#
# Control flags (env):
# - STOP_AFTER_MVTEC_EVAL=1 : stop after mvtec in-domain eval (before training visa)

TAG="${1:?tag required}"
DEVICE="${2:?device_id required}"
shift 2

TRAIN_EXTRA=()
TEST_EXTRA=()
MODE="train"
for arg in "$@"; do
  if [[ "${arg}" == "--train" ]]; then
    MODE="train"
    continue
  fi
  if [[ "${arg}" == "--test" ]]; then
    MODE="test"
    continue
  fi
  if [[ "${MODE}" == "train" ]]; then
    TRAIN_EXTRA+=("${arg}")
  else
    TEST_EXTRA+=("${arg}")
  fi
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRANE_DIR="${ROOT_DIR}/Crane-main"

EPOCHS="${EPOCHS:-5}"

cd "${CRANE_DIR}"

TRAIN_COMMON=(
  --device "${DEVICE}"
  --epoch "${EPOCHS}"
  --save_freq 1
  --train_good_only True
)

TEST_COMMON=(
  --devices "${DEVICE}"
  --epoch "${EPOCHS}"
  --visualize False
)

python3 train.py --dataset mvtec --model_name "${TAG}" "${TRAIN_COMMON[@]}" "${TRAIN_EXTRA[@]}"

# In-domain eval on mvtec (after mvtec training)
python3 test.py --dataset mvtec --model_name "trained_on_mvtec_${TAG}" "${TEST_COMMON[@]}" "${TEST_EXTRA[@]}"

if [[ "${STOP_AFTER_MVTEC_EVAL:-0}" == "1" ]]; then
  echo "STOP_AFTER_MVTEC_EVAL=1 set; stopping after mvtec in-domain eval."
  exit 0
fi

python3 train.py --dataset visa  --model_name "${TAG}" "${TRAIN_COMMON[@]}" "${TRAIN_EXTRA[@]}"

# In-domain eval on visa (after visa training)
python3 test.py --dataset visa  --model_name "trained_on_visa_${TAG}" "${TEST_COMMON[@]}" "${TEST_EXTRA[@]}"

python3 test.py --dataset visa  --model_name "trained_on_mvtec_${TAG}" "${TEST_COMMON[@]}" "${TEST_EXTRA[@]}"
python3 test.py --dataset mvtec --model_name "trained_on_visa_${TAG}"  "${TEST_COMMON[@]}" "${TEST_EXTRA[@]}"
