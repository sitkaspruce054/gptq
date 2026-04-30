#!/usr/bin/env bash
# setup.sh — install dependencies for the GPTQ experiments in the report.
#
# Sets up a Python venv with everything the 5 driver scripts need:
#   run_percdamp_sweep.py, run_blocksize_sweep.py, run_mse_experiment.py,
#   run_opt125m_results.py, run_mixed_precision_sweep.py
#
# All 5 shell out to opt.py, so they share one dependency set:
#   torch (cu121), transformers, datasets, numpy, accelerate
#
# Prerequisites (Rice NOTS cluster):
#   module load Python/3.11.5-GCCcore-13.2.0
#   module load CUDA/12.1.1
#
# Usage:
#   bash setup.sh
#   source $SHARED_SCRATCH/gptq-env-311/bin/activate

set -uo pipefail

VENV="${SHARED_SCRATCH}/gptq-env-311"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

# Keep pip's wheel cache off the home partition (10 GB quota on NOTS)
export PIP_CACHE_DIR="${SHARED_SCRATCH}/pip-cache"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn() { echo -e "${YELLOW}[ warn]${NC} $*"; }
fail() { echo -e "${RED}[ fail]${NC} $*"; }

# Sanity: Python 3.10+
PY_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
log "Using $(python --version 2>&1) at $(which python)"
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    fail "Python 3.10+ required (got $PY_VER). Load the module first:"
    fail "  module load Python/3.11.5-GCCcore-13.2.0"
    exit 1
fi

# Create venv
if [ -d "$VENV" ]; then
    warn "Venv already exists at $VENV — skipping creation (delete to rebuild)"
else
    log "Creating venv at $VENV ..."
    python -m venv "$VENV"
fi

log "Upgrading pip ..."
"$PIP" install --upgrade pip --quiet

# torch with CUDA 12.1
log "Installing torch (cu121) ..."
"$PIP" install torch --index-url https://download.pytorch.org/whl/cu121 --quiet

TORCH_VER=$("$PY" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "ERROR")
CUDA_OK=$("$PY" -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "ERROR")
if [ "$TORCH_VER" = "ERROR" ]; then
    fail "torch import failed — check CUDA module and index URL"
    exit 1
elif [ "$CUDA_OK" = "True" ]; then
    CUDA_VER=$("$PY" -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "?")
    log "torch $TORCH_VER OK — CUDA $CUDA_VER available"
else
    warn "torch $TORCH_VER OK — CUDA not visible yet (expected on login nodes; fine on GPU nodes)"
fi

# HF / numerical stack
log "Installing numpy, transformers, datasets, accelerate ..."
"$PIP" install numpy 'transformers>=4.35.0,<5.0.0' datasets accelerate --quiet

CORE_ERR=0
for PKG in numpy transformers datasets accelerate; do
    if ! "$PY" -c "import ${PKG}" 2>/dev/null; then
        fail "$PKG import failed"
        CORE_ERR=1
    fi
done
if [ $CORE_ERR -ne 0 ]; then
    fail "Core dep install incomplete — cannot proceed"
    exit 1
fi

NUMPY_VER=$("$PY" -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "?")
TF_VER=$("$PY" -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "?")
DS_VER=$("$PY" -c "import datasets; print(datasets.__version__)" 2>/dev/null || echo "?")
log "Core deps OK — numpy $NUMPY_VER  transformers $TF_VER  datasets $DS_VER"

# Verify dataset-loader patches in datautils.py (PTB and C4 both need fixes
# for current HF datasets versions)
DATAUTILS="$(dirname "$0")/datautils.py"
if [ ! -f "$DATAUTILS" ]; then
    warn "datautils.py not found at $DATAUTILS — skipping fix check"
else
    FIXES_OK=1
    if ! grep -q "trust_remote_code=True" "$DATAUTILS"; then
        warn "datautils.py: missing trust_remote_code=True — ptb eval may fail"
        FIXES_OK=0
    fi
    if ! grep -q "verification_mode='no_checks'" "$DATAUTILS"; then
        warn "datautils.py: missing verification_mode='no_checks' — c4 eval may fail"
        FIXES_OK=0
    fi
    if ! grep -q "'en'" "$DATAUTILS"; then
        warn "datautils.py: C4 config may not be 'en' — c4 eval may fail"
        FIXES_OK=0
    fi
    if [ $FIXES_OK -eq 1 ]; then
        log "datautils.py dataset fixes verified (PTB trust_remote_code, C4 en config)"
    fi
fi

echo ""
log "Setup complete"
echo ""
echo "To activate and run all experiments:"
echo "  module load CUDA/12.1.1"
echo "  source $VENV/bin/activate"
echo "  export HF_HOME=\$SHARED_SCRATCH/hf_cache"
echo "  export HF_DATASETS_CACHE=\$SHARED_SCRATCH/hf_cache/datasets"
echo "  cd ~/gptq/gptq"
echo "  bash run_all.sh"
echo ""
