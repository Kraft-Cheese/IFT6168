#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/QRM"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"

# Default experiment settings (override via environment variables)
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}}"
export DATA_DIR
ALGORITHM="${ALGORITHM:-eqrm}"
STEPS="${STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_FREQ="${EVAL_FREQ:-100}"
SEED="${SEED:-0}"
ERM_PRETRAIN_ITERS="${ERM_PRETRAIN_ITERS:-500}"
ALPHA="${ALPHA:-0.9}"
NUM_TRAIN_ENVS="${NUM_TRAIN_ENVS:-8}"
MIN_FMOW_FREE_GB="${MIN_FMOW_FREE_GB:-120}"

ensure_venv() {
	if [[ ! -d "${VENV_DIR}" ]]; then
		echo "[setup] Creating virtual environment at ${VENV_DIR}"
		python3 -m venv "${VENV_DIR}"
	elif [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
		echo "[setup] ${VENV_DIR} exists but is not a valid venv. Recreating it."
		rm -rf "${VENV_DIR}"
		python3 -m venv "${VENV_DIR}"
	else
		echo "[setup] Found existing virtual environment at ${VENV_DIR}"
	fi

	# shellcheck disable=SC1090
	source "${VENV_DIR}/bin/activate"
}

requirements_missing() {
	python - "$REQ_FILE" <<'PY'
import re
import sys
from importlib import metadata

req_file = sys.argv[1]
missing = []

with open(req_file, "r", encoding="utf-8") as f:
		for raw in f:
				line = raw.split("#", 1)[0].strip()
				if not line:
						continue
				m = re.match(r"^([A-Za-z0-9_.-]+)", line)
				if not m:
						continue
				pkg = m.group(1)
				try:
						metadata.version(pkg)
				except metadata.PackageNotFoundError:
						missing.append(pkg)

if missing:
		print("Missing packages:", ", ".join(missing))
		sys.exit(1)

print("All required packages are present.")
PY
}

install_requirements_if_needed() {
	if requirements_missing; then
		echo "[setup] Requirements already installed in QRM."
	else
		echo "[setup] Installing missing requirements from ${REQ_FILE}"
		python -m pip install --upgrade pip
		python -m pip install -r "${REQ_FILE}"
	fi
}

ensure_dataset( dataset_name ) {
	# Look for folders matching dataset_name (with possible suffixes)
	local found_valid=0
	for d in "${DATA_DIR}"/${dataset_name}*; do
		if [[ -d "$d" ]]; then
			if ls "$d"/RELEASE_*.txt 1> /dev/null 2>&1; then
				echo "[setup] Found valid ${dataset_name} dataset at $d"
				found_valid=1
				break
			else
				echo "[setup] Found folder for ${dataset_name} at $d but no release file. Removing folder and re-downloading."
				rm -rf "$d"
			fi
		fi
	done
	if [[ $found_valid -eq 1 ]]; then
		return 0
	fi
}

ensure_fmow_dataset() {
	local fmow_root="${DATA_DIR}/fmow_v1.1"
	local fmow_archive="${fmow_root}/archive.tar.gz"
	local fmow_metadata="${fmow_root}/rgb_metadata.csv"
	local free_gb
	local available_gb
	free_gb="$(df -Pk "${DATA_DIR}" | awk 'NR==2 {print int($4 / 1024 / 1024)}')"
	available_gb="${free_gb:-0}"

	if [[ "${available_gb}" -lt "${MIN_FMOW_FREE_GB}" ]]; then
		echo "[setup] Not enough free space under ${DATA_DIR}."
		echo "[setup] Need at least ${MIN_FMOW_FREE_GB} GB free; found ${available_gb} GB."
		echo "[setup] Free space before retrying FMOW; the download is large and extraction also needs room."
		return 1
	fi

	if [[ -f "${fmow_metadata}" ]]; then
		echo "[setup] FMOW dataset already prepared at ${fmow_root}"
		return 0
	fi

	if [[ -f "${fmow_archive}" ]]; then
		echo "[setup] Removing partial FMOW archive before retrying download"
		rm -f "${fmow_archive}"
	fi

	echo "[setup] Preparing FMOW once before parallel runs"
	python - <<'PY'
import os
from wilds import get_dataset

data_dir = os.environ["DATA_DIR"]
print(f"Preparing FMOW under {data_dir}...")
get_dataset(dataset="fmow", root_dir=data_dir, download=True)
print("FMOW dataset is ready.")
PY
}

run_experiments_parallel() {
	mkdir -p "${DATA_DIR}" "${OUTPUT_DIR}"

	ensure_fmow_dataset

	echo "[run] Starting fmow_temporal and fmow_geo in parallel"

	(
		cd "${SCRIPT_DIR}"
		python train.py \
			--dataset fmow_temporal \
			--data_dir "${DATA_DIR}" \
			--output_dir "${OUTPUT_DIR}" \
			--algorithm "${ALGORITHM}" \
			--num_train_envs "${NUM_TRAIN_ENVS}" \
			--alpha "${ALPHA}" \
			--erm_pretrain_iters "${ERM_PRETRAIN_ITERS}" \
			--steps "${STEPS}" \
			--batch_size "${BATCH_SIZE}" \
			--eval_freq "${EVAL_FREQ}" \
			--seed "${SEED}"
	) &
	temporal_pid=$!

	(
		cd "${SCRIPT_DIR}"
		python train.py \
			--dataset fmow_geo \
			--data_dir "${DATA_DIR}" \
			--output_dir "${OUTPUT_DIR}" \
			--algorithm "${ALGORITHM}" \
			--alpha "${ALPHA}" \
			--erm_pretrain_iters "${ERM_PRETRAIN_ITERS}" \
			--steps "${STEPS}" \
			--batch_size "${BATCH_SIZE}" \
			--eval_freq "${EVAL_FREQ}" \
			--seed "${SEED}"
	) &
	geo_pid=$!

	set +e
	wait "${temporal_pid}"
	temporal_status=$?
	wait "${geo_pid}"
	geo_status=$?
	set -e

	if [[ ${temporal_status} -ne 0 || ${geo_status} -ne 0 ]]; then
		echo "[run] One or more experiments failed. fmow_temporal=${temporal_status}, fmow_geo=${geo_status}"
		return 1
	fi

	echo "[run] Both experiments completed successfully."
}

main() {
	ensure_venv
	install_requirements_if_needed
	run_experiments_parallel
}

main "$@"
