#!/usr/bin/env bash
# run_all.sh — run all 5 GPTQ experiments and emit CSVs.
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
log()  { echo -e "${GREEN}[run_all]${NC} $*"; }
warn() { echo -e "${YELLOW}[ warn ]${NC} $*"; }
fail() { echo -e "${RED}[ fail]${NC} $*"; }

# Sanity: venv active and torch importable
if ! python -c "import torch" 2>/dev/null; then
    fail "torch not importable — activate the venv first:"
    fail "  source \$SHARED_SCRATCH/gptq-env-311/bin/activate"
    exit 1
fi

# Sanity: GPU visible (warning only — opt.py will fail without one anyway)
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    warn "CUDA not visible — opt.py requires a GPU. Are you on a compute node?"
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
    log "[$desc]  python $script"
    t0=$(date +%s)
    python "$script"
    elapsed=$(( $(date +%s) - t0 ))
    if [ -f "$csv" ]; then
        rows=$(($(wc -l < "$csv") - 1))
        log "  -> $csv ($rows result rows, ${elapsed}s)"
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
