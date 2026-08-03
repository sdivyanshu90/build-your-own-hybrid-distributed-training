#!/usr/bin/env bash
#
# Run the hybrid-parallelism example at a rank count the machine can support.
#
#   ./scripts/run_hybrid_example.sh          # picks 4 ranks, or 8 if configured
#   ./scripts/run_hybrid_example.sh 8        # force 8 ranks
#   RANKS=2 ./scripts/run_hybrid_example.sh  # same, via the environment
#
# The script chooses a configuration whose topology product equals the rank
# count, so it never asks torchrun for a world size the config cannot factor.
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPOSITORY_ROOT}"

RANKS="${1:-${RANKS:-4}}"

# Resolve an interpreter rather than assuming `python` exists.  On a stock
# Debian/Ubuntu (and inside most containers) there is only `python3` unless
# `python-is-python3` is installed, so a bare `python` here fails with exit 127
# and the far-from-obvious message "python: command not found" -- which this
# script hit on the development machine.
PYTHON="${PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
  PYTHON="$(command -v python3 || command -v python || true)"
fi
if [[ -z "${PYTHON}" ]]; then
  echo "error: no python3 interpreter on PATH; set PYTHON=/path/to/python" >&2
  exit 2
fi

GPUS="$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')"
CORES="$(getconf _NPROCESSORS_ONLN)"

# NCCL needs one GPU per rank; with fewer GPUs than ranks the run falls back to
# Gloo on CPU, which is correct but slower.  Say so rather than surprising the
# operator with a warning buried in the log.
if [[ "${GPUS}" -ge "${RANKS}" ]]; then
  BACKEND_NOTE="NCCL on ${RANKS} GPU(s)"
  BACKEND_ARGS=()
else
  BACKEND_NOTE="Gloo on CPU (${GPUS} GPU(s) visible, ${RANKS} ranks requested)"
  BACKEND_ARGS=("--backend" "gloo" "--device" "cpu")
fi

if [[ "${RANKS}" -gt "${CORES}" ]]; then
  echo "warning: ${RANKS} ranks on ${CORES} cores will oversubscribe the CPU" >&2
fi

case "${RANKS}" in
  8) CONFIG="configs/hybrid_8gpu.yaml" ;;
  4) CONFIG="configs/hybrid_4gpu.yaml" ;;
  2) CONFIG="configs/ddp_2gpu.yaml" ;;
  *)
    echo "error: no shipped config for ${RANKS} ranks; use 2, 4 or 8" >&2
    exit 2
    ;;
esac

# One thread per rank: without this each rank starts a thread pool sized to the
# whole machine and the ranks fight each other for cores.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "ranks   : ${RANKS}"
echo "config  : ${CONFIG}"
echo "backend : ${BACKEND_NOTE}"
echo

# --standalone runs a single-node rendezvous on a free port, so no scheduler,
# no etcd and no cluster-specific configuration is involved.
exec torchrun --standalone --nproc-per-node="${RANKS}" \
  examples/train_hybrid.py --config "${CONFIG}" "${BACKEND_ARGS[@]}"
