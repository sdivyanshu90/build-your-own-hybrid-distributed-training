#!/usr/bin/env bash
#
# Run the multi-process test suites, one file at a time.
#
#   ./scripts/run_tests_distributed.sh              # everything that runs on CPU
#   ./scripts/run_tests_distributed.sh unit         # a single group
#   TIMEOUT=300 ./scripts/run_tests_distributed.sh  # override the per-test bound
#
# Why one file per pytest invocation
# ----------------------------------
# pytest already runs tests serially inside one process, so this is not about
# ordering -- it is about *isolation and reporting*.  Each file gets its own
# interpreter, so a suite that leaves a wedged process group behind cannot
# affect the next one, and each file reports its own pass/fail line.
#
# What genuinely must not happen is two pytest *processes* running distributed
# tests at once.  Each spawned rank costs a full `import torch` (~400 MB
# resident), so on a memory-constrained machine concurrent runs swap, and a
# five-minute suite takes twenty and looks like a hang.  Never run this script
# alongside another pytest run or the benchmark.  pytest-xdist is wrong here
# for the same reason.
#
# Every child rank also gets OMP_NUM_THREADS=1 (see conftest.py); without it
# four ranks each start an eight-thread pool on an eight-core machine.
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPOSITORY_ROOT}"

TIMEOUT="${TIMEOUT:-300}"
PYTEST="${PYTEST:-python -m pytest}"
EXTRA_ARGS=("-q" "--timeout=${TIMEOUT}" "-p" "no:cacheprovider")

GROUP="${1:-all}"

declare -a TARGETS
case "${GROUP}" in
  unit)          TARGETS=("tests/unit") ;;
  distributed)   TARGETS=(
                   "tests/distributed/test_collectives.py"
                   "tests/distributed/test_ddp.py"
                   "tests/distributed/test_fsdp.py"
                   "tests/distributed/test_tensor_parallel.py"
                 ) ;;
  integration)   TARGETS=("tests/integration") ;;
  e2e)           TARGETS=("tests/end_to_end") ;;
  performance)   TARGETS=("tests/performance") ;;
  all)           TARGETS=(
                   "tests/unit"
                   "tests/distributed/test_collectives.py"
                   "tests/distributed/test_ddp.py"
                   "tests/distributed/test_fsdp.py"
                   "tests/distributed/test_tensor_parallel.py"
                   "tests/integration"
                   "tests/end_to_end"
                   "tests/performance"
                 ) ;;
  *)
    echo "usage: $0 [unit|distributed|integration|e2e|performance|all]" >&2
    exit 2
    ;;
esac

echo "repository : ${REPOSITORY_ROOT}"
echo "python     : $(python --version 2>&1)"
echo "torch      : $(python -c 'import torch; print(torch.__version__)')"
echo "cuda       : $(python -c 'import torch; print(torch.cuda.device_count())') device(s)"
echo "cores      : $(getconf _NPROCESSORS_ONLN)"
echo "per-test timeout: ${TIMEOUT}s"
echo

FAILED=0
for target in "${TARGETS[@]}"; do
  echo "=============================================================="
  echo ">>> ${target}"
  echo "=============================================================="
  if ! ${PYTEST} "${EXTRA_ARGS[@]}" -m "not cuda and not multigpu" "${target}"; then
    echo "!!! FAILED: ${target}" >&2
    FAILED=1
  fi
  echo
done

if [[ "${FAILED}" -ne 0 ]]; then
  echo "one or more test groups failed" >&2
  exit 1
fi
echo "all requested test groups passed"
