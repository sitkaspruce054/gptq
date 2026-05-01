#!/usr/bin/env bash
# run_all.sh - run all 5 GPTQ experiments and emit CSVs.
#
# Assumes setup.sh has been run and the venv is activated.
# Each driver writes its CSV in the current directory.
#
# Outputs:
#   results_opt125m.csv     paper reproduction (Tables 3 / 9 / 11)
#   results_percdamp.csv    Hessian dampening sweep
#   results_blocksize.csv   block-size sweep
#   results_mse.csv         min-max vs MSE clipping
#   results_mixed.csv       mixed-precision sweep

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log() { echo -e "${GREEN}[run_all]${NC} $*"; }
warn() { echo -e "${YELLOW}[ warn ]${NC} $*"; }
fail() { echo -e "${RED}[ fail]${NC} $*"; }

#needed for NOTS space
if [ -n "${SHARED_SCRATCH:-}" ]; then
    : "${HF_HOME:=$SHARED_SCRATCH/hf_cache}"
    : "${HF_DATASETS_CACHE:=$SHARED_SCRATCH/hf_cache/datasets}"
    export HF_HOME HF_DATASETS_CACHE
    mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"
    log "HF_HOME=$HF_HOME"
    log "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
else
    warn "SHARED_SCRATCH not set - HF will use ~/.cache/huggingface (may hit quota)"
fi

#checking imports
if ! python -c "import torch" 2>/dev/null; then
    fail "torch not importable - activate the venv first:"
    fail "  source \$SHARED_SCRATCH/gptq-env-311/bin/activate"
    exit 1
fi

#make sure gpu is accessible
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    warn "CUDA not visible"
fi

# Sanity: HF cache target is writable and not on a near-full filesystem
if [ -n "${HF_HOME:-}" ]; then
    if ! touch "$HF_HOME/.write_test" 2>/dev/null; then
        fail "HF_HOME=$HF_HOME is not writable"
        exit 1
    fi
    rm -f "$HF_HOME/.write_test"
    AVAIL_KB=$(df -P "$HF_HOME" 2>/dev/null | awk 'NR==2 {print $4}')
    if [ -n "$AVAIL_KB" ] && [ "$AVAIL_KB" -lt 1048576 ]; then
        warn "HF_HOME has <1 GB free - model downloads may fail"
    fi
fi

EXPERIMENTS=(
    "run_opt125m_results.py:results_opt125m.csv:paper reproduction"
    "run_percdamp_sweep.py:results_percdamp.csv:Hessian dampening sweep"
    "run_blocksize_sweep.py:results_blocksize.csv:block-size sweep"
    "run_mse_experiment.py:results_mse.csv:min-max vs MSE clipping"
    "run_mixed_precision_sweep.py:results_mixed.csv:mixed-precision sweep"
)

T0=$(date +%s)
for entry in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r script csv desc <<< "$entry"
    log ""
    log "[$desc] python $script"
    t0=$(date +%s)
    python "$script"
    elapsed=$(( $(date +%s) - t0 ))
    if [ -f "$csv" ]; then
        rows=$(($(wc -l < "$csv") - 1))
        log " -> $csv ($rows result rows, ${elapsed}s)"
    else
        fail "$script did not produce $csv"
        exit 1
    fi
done

TOTAL=$(( $(date +%s) - T0 ))
log ""
log "All 5 experiments complete in ${TOTAL}s"
log "CSVs:"
ls -la results_*.csv
