#!/bin/bash
# PovertyMap experiment runner
# Usage:
#   ./run_poverty.sh                          full 200-epoch comparison (ERM/GroupDRO/VREx/EQRM)
#   ./run_poverty.sh --quick                  20 epochs, 20% data (sanity-check)
#   ./run_poverty.sh --quantile               EQRM α ablation (0.5 / 0.75 / 0.9)
#   ./run_poverty.sh --no-wandb               disable W&B logging
#   ./run_poverty.sh --project my-project     override W&B project name

set -euo pipefail

ROOT_DIR="/Volumes/WILDS_DATA/wilds_data"
LOG_BASE="$(cd "$(dirname "$0")" && pwd)/wilds_logs/poverty"
EXAMPLES="$(cd "$(dirname "$0")" && pwd)/examples"

N_EPOCHS=200
FRAC=1.0
SEED=0
MODE="full"
USE_WANDB=true
WANDB_PROJECT="IFT6168-poverty"

# Parse flags
while [[ $# -gt 0 ]]; do
  case $1 in
    --quick)       N_EPOCHS=20;   FRAC=0.2;  MODE="quick"    ;;
    --quantile)                              MODE="quantile"  ;;
    --no-wandb)    USE_WANDB=false                            ;;
    --project)     WANDB_PROJECT="$2"; shift                  ;;
    --seed)        SEED="$2";     shift                       ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
  shift
done

mkdir -p "$LOG_BASE"

# run ALGO [QUANTILE]
#   ALGO     one of ERM groupDRO VREx EQRM
#   QUANTILE optional float, only used with EQRM; omit to use default (0.75)
run() {
  local ALGO="$1"
  local QUANTILE="${2:-}"

  # Build identifier for run
  local RUN_ID="poverty_${ALGO}_seed${SEED}"
  [[ -n "$QUANTILE" ]] && RUN_ID="poverty_${ALGO}_q${QUANTILE}_seed${SEED}"
  [[ "$MODE" == "quick"    ]] && RUN_ID="${RUN_ID}_quick"

  local DIR="$LOG_BASE/$RUN_ID"
  mkdir -p "$DIR"

  echo ""
  echo " dataset : poverty"
  echo " algo    : ${ALGO}${QUANTILE:+ (α=$QUANTILE)}"
  echo " epochs  : ${N_EPOCHS}  frac: ${FRAC}  seed: ${SEED}"
  [[ "$USE_WANDB" == true ]] && \
  echo " wandb   : ${WANDB_PROJECT} / ${RUN_ID}"
  echo " log dir : ${DIR}"

  # Build optional args array
  local EXTRA=()
  [[ -n "$QUANTILE" ]] && EXTRA+=(--var_quantile "$QUANTILE")

  if [[ "$USE_WANDB" == true ]]; then
    EXTRA+=(
      --use_wandb True
      --wandb_kwargs project="$WANDB_PROJECT" name="$RUN_ID"
    )
  fi

  cd "$EXAMPLES"
  python run_expt.py \
    -d poverty \
    --algorithm "$ALGO" \
    --root_dir "$ROOT_DIR" \
    --download True \
    --seed "$SEED" \
    --n_epochs "$N_EPOCHS" \
    --frac "$FRAC" \
    --log_dir "$DIR" \
    --progress_bar True \
    --save_best True \
    --save_last True \
    ${EXTRA[@]+"${EXTRA[@]}"} \
    2>&1 | tee "$DIR/run.log"
}

# Experiment plans
if [[ "$MODE" == "quantile" ]]; then
  # EQRM α ablation keeps default run (0.75) plus two neighbours
  for Q in 0.5 0.75 0.9; do
    run EQRM "$Q"
  done
else
  # Main four-way comparison
  for ALGO in ERM groupDRO IRM EQRM; do
    run "$ALGO"
  done
fi

echo ""
echo "Done. Results in $LOG_BASE"
[[ "$USE_WANDB" == true ]] && \
echo "W&B project: https://wandb.ai/$(wandb status 2>/dev/null | grep 'Entity' | awk '{print $2}')/${WANDB_PROJECT}" || true
