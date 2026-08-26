#!/bin/bash
#SBATCH --job-name=e012_radial_localization
#SBATCH --output=/home/xggh8/data/zw-lab/slurm-e012-localization-%j.out
#SBATCH --error=/home/xggh8/data/zw-lab/slurm-e012-localization-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "error: SLURM_JOB_ID is unset" >&2
  exit 1
fi
if [ -z "${E012_REPO_ROOT:-}" ] || [ -z "${E012_REPO_COMMIT:-}" ]; then
  echo "error: E012_REPO_ROOT and E012_REPO_COMMIT are required" >&2
  exit 1
fi
case "$E012_REPO_ROOT" in
  /*) ;;
  *) echo "error: E012_REPO_ROOT must be absolute" >&2; exit 1 ;;
esac

REPO_ROOT="$(cd "$E012_REPO_ROOT" && pwd -P)"
ENTRYPOINT="${REPO_ROOT}/experiments/12_frequency_localization.py"
CONFIG="${REPO_ROOT}/configs/e012_frequency_localization.json"
if [ ! -d "${REPO_ROOT}/.git" ] || [ ! -f "$ENTRYPOINT" ] || [ ! -f "$CONFIG" ]; then
  echo "error: E012 checkout is incomplete: ${REPO_ROOT}" >&2
  exit 1
fi
if [ "$(git -C "$REPO_ROOT" rev-parse HEAD)" != "$E012_REPO_COMMIT" ]; then
  echo "error: E012 repository commit mismatch" >&2
  exit 1
fi
if [ -n "$(git -C "$REPO_ROOT" status --porcelain=v1)" ]; then
  echo "error: E012 repository worktree is not clean" >&2
  exit 1
fi

MODE="${1:-}"
case "$MODE" in
  preflight) OUTPUT_DIR="" ;;
  smoke) OUTPUT_DIR="/home/xggh8/scratch/zw-lab/e012_frequency_localization_smoke_${SLURM_JOB_ID}" ;;
  full) OUTPUT_DIR="/home/xggh8/data/zw-lab/e012_frequency_localization" ;;
  resume) OUTPUT_DIR="/home/xggh8/data/zw-lab/e012_frequency_localization" ;;
  *) echo "error: usage: sbatch $0 {preflight|smoke|full|resume}" >&2; exit 1 ;;
esac
if [ -n "$OUTPUT_DIR" ] && [ -e "$OUTPUT_DIR" ] && [ "$MODE" != "resume" ]; then
  echo "error: output path already exists: ${OUTPUT_DIR}" >&2
  exit 1
fi

if [ -n "${SLURM_TMPDIR:-}" ]; then
  TMP_ROOT="$SLURM_TMPDIR"
  TMP_SOURCE="provided"
else
  TMP_ROOT="/cluster/pixstor/zwggh-lab/xinyu/slurm_tmp/e012_${SLURM_JOB_ID}"
  TMP_SOURCE="fallback"
fi
mkdir -p "$TMP_ROOT"
if [ ! -d "$TMP_ROOT" ] || [ ! -w "$TMP_ROOT" ]; then
  echo "error: temporary directory is not writable: ${TMP_ROOT}" >&2
  exit 1
fi
export SLURM_TMPDIR="$TMP_ROOT"
export TMPDIR="${TMP_ROOT}/tmp"
export XDG_CACHE_HOME="${TMP_ROOT}/cache"
export TORCH_HOME="${TMP_ROOT}/torch_cache"
export MPLCONFIGDIR="${TMP_ROOT}/matplotlib"
export WANDB_DIR="${TMP_ROOT}/wandb"
export HF_HOME="${TMP_ROOT}/hf"
export TRANSFORMERS_CACHE="${TMP_ROOT}/hf/transformers"
export HF_DATASETS_CACHE="${TMP_ROOT}/hf/datasets"
export PIP_CACHE_DIR="${TMP_ROOT}/pip-cache"
export DNNLIB_CACHE_DIR="${TMP_ROOT}/dnnlib"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${REPO_ROOT}/src:/home/xggh8/edm:${PYTHONPATH:-}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" \
  "$WANDB_DIR" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" \
  "$PIP_CACHE_DIR" "$DNNLIB_CACHE_DIR"

echo "repository_root=${REPO_ROOT}"
echo "repository_commit=${E012_REPO_COMMIT}"
echo "mode=${MODE}"
echo "output_dir=${OUTPUT_DIR:-none}"
echo "slurm_tmpdir_source=${TMP_SOURCE}"
echo "slurm_tmpdir=${SLURM_TMPDIR}"
echo "training_executed=false"

python - <<'PY'
import sys
import torch
print(f"python_executable={sys.executable}")
print(f"torch_version={torch.__version__}")
print(f"torch_file={torch.__file__}")
PY
python -m torch.distributed.run --help >/dev/null
echo "torch_distributed_run_help=pass"

cd "$REPO_ROOT"
if [ "$MODE" = "preflight" ]; then
  python "$ENTRYPOINT" --mode preflight --config "$CONFIG"
elif [ "$MODE" = "resume" ]; then
  python "$ENTRYPOINT" --mode full --resume --config "$CONFIG" --output-dir "$OUTPUT_DIR"
else
  python "$ENTRYPOINT" --mode "$MODE" --config "$CONFIG" --output-dir "$OUTPUT_DIR"
fi
