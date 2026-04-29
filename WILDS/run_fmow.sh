#!/bin/bash
# FMoW experiment runner
# Usage:
#   ./run_fmow.sh                         full 30-epoch comparison (ERM/GroupDRO/VREx/EQRM)
#   ./run_fmow.sh --quick                 3 epochs, 10% data (sanity-check)
#   ./run_fmow.sh --quantile              EQRM α ablation (0.5 / 0.75 / 0.9)
#   ./run_fmow.sh --poison year_tint      run all 4 algos with year_tint poison applied
#   ./run_fmow.sh --poison temporal_gap run all 4 algos with temporal_gap (middle environment has added noise)
#   ./run_fmow.sh --no-wandb              disable W&B logging
#   ./run_fmow.sh --project my-project    override W&B project name

set -euo pipefail

ROOT_DIR="/data/rech/hamelcas/wilds_data"
LOG_BASE="$(cd "$(dirname "$0")" && pwd)/wilds_logs/fmow"
EXAMPLES="$(cd "$(dirname "$0")" && pwd)/examples"

N_EPOCHS=30
FRAC=1.0
SEED=0
MODE="full"
POISON_MODE=""
USE_WANDB=true
WANDB_PROJECT="IFT6168-fmow"

# Parse flags
while [[ $# -gt 0 ]]; do
  case $1 in
    --quick)       N_EPOCHS=3;    FRAC=0.1;  MODE="quick"    ;;
    --quantile)                              MODE="quantile"  ;;
    --poison)      POISON_MODE="$2"; shift                   ;;
    --no-wandb)    USE_WANDB=false                           ;;
    --project)     WANDB_PROJECT="$2"; shift                 ;;
    --seed)        SEED="$2";     shift                      ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
  shift
done

mkdir -p "$LOG_BASE"

# run ALGO [QUANTILE]
run() {
  local ALGO="$1"
  local QUANTILE="${2:-}"

  local RUN_ID="fmow_${ALGO}_seed${SEED}"
  [[ -n "$QUANTILE"    ]] && RUN_ID="fmow_${ALGO}_q${QUANTILE}_seed${SEED}"
  [[ -n "$POISON_MODE" ]] && RUN_ID="${RUN_ID}_poison_${POISON_MODE}"
  [[ "$MODE" == "quick" ]] && RUN_ID="${RUN_ID}_quick"

  local DIR="$LOG_BASE/$RUN_ID"
  mkdir -p "$DIR"

  echo ""
  echo " dataset : fmow"
  echo " algo    : ${ALGO}${QUANTILE:+ (α=$QUANTILE)}"
  echo " epochs  : ${N_EPOCHS}  frac: ${FRAC}  seed: ${SEED}"
  [[ -n "$POISON_MODE" ]] && echo " poison  : ${POISON_MODE}"
  [[ "$USE_WANDB" == true ]] && \
  echo " wandb   : ${WANDB_PROJECT} / ${RUN_ID}"
  echo " log dir : ${DIR}"

  local EXTRA=()
  [[ -n "$QUANTILE"    ]] && EXTRA+=(--var_quantile "$QUANTILE")
  [[ -n "$POISON_MODE" ]] && EXTRA+=(--poison_mode "$POISON_MODE")

  if [[ "$USE_WANDB" == true ]]; then
    EXTRA+=(
      --use_wandb True
      --wandb_kwargs project="$WANDB_PROJECT" name="$RUN_ID"
    )
  fi

  cd "$EXAMPLES"
  python run_expt.py \
    -d fmow \
    --algorithm "$ALGO" \
    --root_dir "$ROOT_DIR" \
    --seed "$SEED" \
    --n_epochs "$N_EPOCHS" \
    --frac "$FRAC" \
    --log_dir "$DIR" \
    --progress_bar True \
    --save_best True \
    --save_last False \
    ${EXTRA[@]+"${EXTRA[@]}"} \
    2>&1 | tee "$DIR/run.log"
}

# Experiment plans
if [[ "$MODE" == "quantile" ]]; then
  for Q in 0.5 0.75 0.9; do
    run EQRM "$Q"
  done
else
  for ALGO in ERM groupDRO VREx EQRM; do
    run "$ALGO"
  done
fi

echo ""
echo "Done. Results in $LOG_BASE"
