#!/bin/bash
# FMoW temporal experiment runner (WildTime)
# Usage:
#   ./run_fmow.sh                         full comparison (ERM/GroupDRO/VREx/EQRM), 5000 steps
#   ./run_fmow.sh --quick                 500 steps, sanity-check
#   ./run_fmow.sh --quantile              EQRM alpha ablation (q=0.5 / 0.75 / 0.9)
#   ./run_fmow.sh --poison year_tint      all 4 algos with year_tint poison
#   ./run_fmow.sh --poison temporal_gap   all 4 algos with temporal_gap poison
#   ./run_fmow.sh --no-wandb              disable W&B logging
#   ./run_fmow.sh --project my-project    override W&B project name

set -euo pipefail

WILDTIME="$(cd "$(dirname "$0")/../WildTime" && pwd)"
DATA_DIR="/data/rech/hamelcas/wilds_data"
OUTPUT_DIR="$WILDTIME"

STEPS=5000
ERM_PRETRAIN=500
NUM_ENVS=8
SEED=0
MODE="full"
POISON_MODE=""
USE_WANDB=true
WANDB_PROJECT="IFT6168-fmow-temporal"

# Parse flags
while [[ $# -gt 0 ]]; do
  case $1 in
    --quick)       STEPS=500;  ERM_PRETRAIN=50;  MODE="quick"    ;;
    --quantile)                                  MODE="quantile"  ;;
    --poison)      POISON_MODE="$2"; shift                       ;;
    --no-wandb)    USE_WANDB=false                               ;;
    --project)     WANDB_PROJECT="$2"; shift                     ;;
    --seed)        SEED="$2";  shift                             ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
  shift
done

# run ALGO [ALPHA]
#   ALGO   one of erm groupdro vrex eqrm
#   ALPHA  optional log-space quantile for EQRM ablation (empty = default -500 ~ q=1.0)
run() {
  local ALGO="$1"
  local ALPHA="${2:-}"

  local RUN_ID="fmow_${ALGO}_seed${SEED}"
  [[ -n "$ALPHA"        ]] && RUN_ID="fmow_${ALGO}_alpha${ALPHA}_seed${SEED}"
  [[ -n "$POISON_MODE"  ]] && RUN_ID="${RUN_ID}_poison_${POISON_MODE}"
  [[ "$MODE" == "quick" ]] && RUN_ID="${RUN_ID}_quick"

  local LOG_DIR="${OUTPUT_DIR}/logs/${RUN_ID}"
  mkdir -p "$LOG_DIR"

  echo ""
  echo " dataset : fmow_temporal (WildTime)"
  echo " algo    : ${ALGO}${ALPHA:+ (alpha=$ALPHA)}"
  echo " steps   : ${STEPS}  pretrain: ${ERM_PRETRAIN}  envs: ${NUM_ENVS}  seed: ${SEED}"
  [[ -n "$POISON_MODE" ]] && echo " poison  : ${POISON_MODE}"
  [[ "$USE_WANDB" == true ]] && \
  echo " wandb   : ${WANDB_PROJECT} / ${RUN_ID}"

  local EXTRA=()
  [[ -n "$ALPHA"       ]] && EXTRA+=(--alpha "$ALPHA")
  [[ -n "$POISON_MODE" ]] && EXTRA+=(--poison_mode "$POISON_MODE")
  if [[ "$USE_WANDB" == true ]]; then
    EXTRA+=(--use_wandb --wandb_project "$WANDB_PROJECT" --wandb_run_name "$RUN_ID")
  fi

  cd "$WILDTIME"
  python train.py \
    --dataset fmow_temporal \
    --algorithm "$ALGO" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --exp_name "$RUN_ID" \
    --num_train_envs "$NUM_ENVS" \
    --steps "$STEPS" \
    --erm_pretrain_iters "$ERM_PRETRAIN" \
    --seed "$SEED" \
    ${EXTRA[@]+"${EXTRA[@]}"} \
    2>&1 | tee "$LOG_DIR/run.log"
}

# Experiment plans
if [[ "$MODE" == "quantile" ]]; then
  # Alpha encodes log(1-q): q=0.5 -0.693, q=0.75 -1.386, q=0.9 -2.303
  for ALPHA in -0.693 -1.386 -2.303; do
    run eqrm "$ALPHA"
  done
else
  for ALGO in erm groupdro vrex eqrm; do
    run "$ALGO"
  done
fi

echo ""
echo "Done. Results in ${OUTPUT_DIR}/results/"
