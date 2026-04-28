#!/bin/bash
# CivilComments experiment runner
# Usage:
#   ./run_civilcomments.sh                        full 5-epoch comparison (ERM/GroupDRO/VREx/EQRM)
#   ./run_civilcomments.sh --quick                1 epoch, 10% data (sanity-check)
#   ./run_civilcomments.sh --no-wandb             disable W&B logging
#   ./run_civilcomments.sh --project my-project   override W&B project name

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)/wilds_data"
LOG_BASE="$(cd "$(dirname "$0")" && pwd)/wilds_logs/civilcomments"
EXAMPLES="$(cd "$(dirname "$0")" && pwd)/examples"

N_EPOCHS=5
FRAC=1.0
SEED=0
MODE="full"
USE_WANDB=true
WANDB_PROJECT="IFT6168-civilcomments"

# Parse flags
while [[ $# -gt 0 ]]; do
  case $1 in
    --quick)       N_EPOCHS=1;   FRAC=0.1;  MODE="quick"    ;;
    --no-wandb)    USE_WANDB=false                           ;;
    --project)     WANDB_PROJECT="$2"; shift                 ;;
    --seed)        SEED="$2";    shift                       ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
  shift
done

mkdir -p "$LOG_BASE"

# run ALGO [QUANTILE]
run() {
  local ALGO="$1"
  local QUANTILE="${2:-}"

  local RUN_ID="civilcomments_${ALGO}_seed${SEED}"
  [[ -n "$QUANTILE" ]] && RUN_ID="civilcomments_${ALGO}_q${QUANTILE}_seed${SEED}"
  [[ "$MODE" == "quick"    ]] && RUN_ID="${RUN_ID}_quick"

  local DIR="$LOG_BASE/$RUN_ID"
  mkdir -p "$DIR"

  echo ""
  echo " dataset : civilcomments"
  echo " algo    : ${ALGO}${QUANTILE:+ (α=$QUANTILE)}"
  echo " epochs  : ${N_EPOCHS}  frac: ${FRAC}  seed: ${SEED}"
  [[ "$USE_WANDB" == true ]] && \
  echo " wandb   : ${WANDB_PROJECT} / ${RUN_ID}"
  echo " log dir : ${DIR}"

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
    -d civilcomments \
    --algorithm "$ALGO" \
    --root_dir "$ROOT_DIR" \
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
for ALGO in ERM groupDRO IRM EQRM; do
  run "$ALGO"
done

echo ""
echo "Done. Results in $LOG_BASE"
