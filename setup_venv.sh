#!/usr/bin/env bash
# setup_venv.sh — create a Python 3.11 venv and install benchmark deps.
#
# Usage:
#   module load CUDA/12.1.1
#   bash setup_venv.sh
#   source $SCRATCH/gptq-env-311/bin/activate
#
# After activation, run:
#   python run_quant_comparison.py --model TinyLlama-1.1B-Chat-v1.0

set -uo pipefail

VENV="${SCRATCH}/gptq-env-311"
GPTQMODEL_DIR="${HOME}/gptq/GPTQModel"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn() { echo -e "${YELLOW}[ warn]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC}  $*"; }

# Track optional dep results for summary
declare -A STATUS

# ── 0. sanity: need Python 3.10+ to satisfy GPTQModel ────────────────────────
PY_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
log "Using $(python --version 2>&1) at $(which python)"
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    fail "Python 3.10+ required (got $PY_VER). Load Python/3.11.5 module first:"
    fail "  module load Python/3.11.5-GCCcore-13.2.0"
    exit 1
fi

# ── 1. create venv ────────────────────────────────────────────────────────────
if [ -d "$VENV" ]; then
    warn "Venv already exists at $VENV — skipping creation (delete it to rebuild)"
else
    log "Creating venv at $VENV ..."
    python -m venv "$VENV"
fi

log "Upgrading pip ..."
"$PIP" install --upgrade pip --quiet

# ── 2. torch (CUDA 12.1) ─────────────────────────────────────────────────────
log "Installing torch (cu121) ..."
"$PIP" install torch --index-url https://download.pytorch.org/whl/cu121 --quiet

CUDA_OK=$("$PY" -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "ERROR")
if [ "$CUDA_OK" = "True" ]; then
    log "torch OK — CUDA available"
elif [ "$CUDA_OK" = "False" ]; then
    warn "torch OK — CUDA not visible yet (expected on login nodes; fine on GPU nodes)"
else
    fail "torch import failed — check CUDA module and index URL"
    exit 1
fi

# ── 3. HF stack ──────────────────────────────────────────────────────────────
log "Installing transformers, datasets, sentencepiece, accelerate ..."
"$PIP" install transformers datasets sentencepiece accelerate --quiet
log "HF stack OK"

# ── 4. bitsandbytes ──────────────────────────────────────────────────────────
log "Installing bitsandbytes ..."
if "$PIP" install bitsandbytes --quiet 2>/dev/null; then
    if "$PY" -c "import bitsandbytes" 2>/dev/null; then
        log "bitsandbytes OK"
        STATUS[bnb]="ok"
    else
        warn "bitsandbytes installed but import fails — triton compile issue"
        warn "Trying pinned version without triton dependency ..."
        "$PIP" install 'bitsandbytes==0.41.3' --quiet 2>/dev/null || true
        if "$PY" -c "import bitsandbytes" 2>/dev/null; then
            log "bitsandbytes 0.41.3 OK"
            STATUS[bnb]="ok (0.41.3)"
        else
            warn "bitsandbytes still broken — bnb_nf4/bnb_int8 will be skipped"
            STATUS[bnb]="SKIP"
        fi
    fi
else
    warn "bitsandbytes install failed — bnb_nf4/bnb_int8 will be skipped"
    STATUS[bnb]="SKIP"
fi

# ── 5. autoawq ───────────────────────────────────────────────────────────────
log "Installing autoawq ..."
if "$PIP" install autoawq --quiet 2>/dev/null; then
    if "$PY" -c "import awq" 2>/dev/null; then
        log "autoawq OK"
        STATUS[awq]="ok"
    else
        warn "autoawq installed but import fails — awq_4bit will be skipped"
        STATUS[awq]="SKIP"
    fi
else
    warn "autoawq install failed — awq_4bit will be skipped"
    STATUS[awq]="SKIP"
fi

# ── 6. GPTQModel (no-deps first, then fill gaps) ──────────────────────────────
if [ -d "$GPTQMODEL_DIR" ]; then
    log "Installing GPTQModel from $GPTQMODEL_DIR (no-deps) ..."
    if "$PIP" install -e "$GPTQMODEL_DIR" --no-deps --quiet 2>/dev/null; then
        IMPORT_ERR=$("$PY" -c "import gptqmodel" 2>&1 || true)
        if [ -z "$IMPORT_ERR" ]; then
            log "gptqmodel OK"
            STATUS[gptqmodel]="ok"
        else
            # Extract the first missing module name and try to install it
            MISSING=$(echo "$IMPORT_ERR" | grep -oP "No module named '\K[^']+" | head -1)
            if [ -n "$MISSING" ]; then
                warn "gptqmodel missing dep: $MISSING — trying to install ..."
                "$PIP" install "$MISSING" --quiet 2>/dev/null || true
                if "$PY" -c "import gptqmodel" 2>/dev/null; then
                    log "gptqmodel OK (after installing $MISSING)"
                    STATUS[gptqmodel]="ok"
                else
                    warn "gptqmodel still broken after installing $MISSING"
                    warn "Run: $PIP install -e $GPTQMODEL_DIR  (with deps) to resolve"
                    STATUS[gptqmodel]="SKIP"
                fi
            else
                warn "gptqmodel import error: $IMPORT_ERR"
                STATUS[gptqmodel]="SKIP"
            fi
        fi
    else
        warn "GPTQModel install failed — gptqmodel_* methods will be skipped"
        STATUS[gptqmodel]="SKIP"
    fi
else
    warn "GPTQModel not found at $GPTQMODEL_DIR — skipping"
    STATUS[gptqmodel]="SKIP (not found)"
fi

# ── summary ───────────────────────────────────────────────────────────────────
echo ""
log "══════════════════════════════════════"
log " Setup complete"
log "══════════════════════════════════════"
echo ""
echo "  bitsandbytes : ${STATUS[bnb]}"
echo "  autoawq      : ${STATUS[awq]}"
echo "  gptqmodel    : ${STATUS[gptqmodel]}"
echo ""
echo "To use this environment:"
echo "  source $VENV/bin/activate"
echo ""
echo "Then run the benchmark:"
echo "  cd ~/gptq/gptq"
echo "  python run_quant_comparison.py --model TinyLlama-1.1B-Chat-v1.0"
