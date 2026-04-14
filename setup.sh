#!/usr/bin/env bash
# setup.sh — install and verify all dependencies for GPTQ experiments.
#
# Covers:
#   Core (required for every script in this repo):
#     torch (cu121), transformers, datasets, sentencepiece, accelerate, numpy
#   Optional (run_quant_comparison.py only):
#     bitsandbytes  — bnb_nf4 / bnb_int8 methods
#     autoawq       — awq_4bit method (deprecated upstream, may fail on newer transformers)
#     gptqmodel     — gptqmodel_* methods; requires ~/gptq/GPTQModel repo
#
# Prerequisites:
#   module load Python/3.11.5-GCCcore-13.2.0
#   module load CUDA/12.1.1
#
# Usage:
#   bash setup.sh
#   source $SHARED_SCRATCH/gptq-env-311/bin/activate

set -uo pipefail

VENV="${SHARED_SCRATCH}/gptq-env-311"
GPTQMODEL_DIR="${HOME}/gptq/GPTQModel"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

# Keep pip's wheel cache off the home partition (10 GB quota)
export PIP_CACHE_DIR="${SHARED_SCRATCH}/pip-cache"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn() { echo -e "${YELLOW}[ warn]${NC} $*"; }
fail() { echo -e "${RED}[ fail]${NC} $*"; }

declare -A STATUS

# ── 0. Python version ─────────────────────────────────────────────────────────
PY_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
log "Using $(python --version 2>&1) at $(which python)"
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    fail "Python 3.10+ required (got $PY_VER). Load the module first:"
    fail "  module load Python/3.11.5-GCCcore-13.2.0"
    exit 1
fi

# ── 1. create venv ────────────────────────────────────────────────────────────
if [ -d "$VENV" ]; then
    warn "Venv already exists at $VENV — skipping creation (delete to rebuild)"
else
    log "Creating venv at $VENV ..."
    python -m venv "$VENV"
fi

log "Upgrading pip ..."
"$PIP" install --upgrade pip --quiet

# ── 2. torch (CUDA 12.1) ─────────────────────────────────────────────────────
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
    STATUS[torch]="$TORCH_VER (CUDA $CUDA_VER)"
else
    warn "torch $TORCH_VER OK — CUDA not visible yet (expected on login nodes; fine on GPU nodes)"
    STATUS[torch]="$TORCH_VER (CUDA not checked — login node)"
fi

# ── 3. core deps ──────────────────────────────────────────────────────────────
# numpy:       datautils.py imports it at module level — required for all scripts
# transformers: model loading, tokenizers
# datasets:    wikitext2, ptb, c4 loading
# sentencepiece: tokenizer backend for OPT/LLaMA models
# accelerate:  HuggingFace model dispatch utilities
log "Installing numpy, transformers, datasets, sentencepiece, accelerate ..."
"$PIP" install numpy transformers datasets sentencepiece accelerate --quiet

CORE_ERR=0
for PKG in numpy transformers datasets sentencepiece accelerate; do
    if ! "$PY" -c "import ${PKG//-/_}" 2>/dev/null; then
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
STATUS[core]="ok (numpy $NUMPY_VER, transformers $TF_VER, datasets $DS_VER)"

# ── 4. verify datautils.py dataset fixes ─────────────────────────────────────
DATAUTILS="$(dirname "$0")/datautils.py"
if [ ! -f "$DATAUTILS" ]; then
    warn "datautils.py not found at $DATAUTILS — skipping fix check"
    STATUS[datautils]="not found"
else
    FIXES_OK=1
    if ! grep -q "trust_remote_code=True" "$DATAUTILS"; then
        warn "datautils.py: missing trust_remote_code=True for PTB — ptb eval may fail"
        FIXES_OK=0
    fi
    if ! grep -q "verification_mode='no_checks'" "$DATAUTILS"; then
        warn "datautils.py: missing verification_mode='no_checks' for C4 — c4 eval may fail"
        FIXES_OK=0
    fi
    if ! grep -q "'en'" "$DATAUTILS"; then
        warn "datautils.py: C4 config may not be 'en' — c4 eval may fail"
        FIXES_OK=0
    fi
    if [ $FIXES_OK -eq 1 ]; then
        log "datautils.py dataset fixes verified (PTB trust_remote_code, C4 en config)"
        STATUS[datautils]="ok"
    else
        STATUS[datautils]="WARN — see above"
    fi
fi

# ── 5. bitsandbytes (optional) ───────────────────────────────────────────────
log "Installing bitsandbytes ..."
if "$PIP" install bitsandbytes --quiet 2>/dev/null; then
    if "$PY" -c "import bitsandbytes" 2>/dev/null; then
        BNB_VER=$("$PY" -c "import bitsandbytes; print(bitsandbytes.__version__)" 2>/dev/null || echo "?")
        log "bitsandbytes $BNB_VER OK"
        STATUS[bnb]="ok ($BNB_VER)"
    else
        warn "bitsandbytes installed but import fails — trying pinned 0.41.3 ..."
        "$PIP" install 'bitsandbytes==0.41.3' --quiet 2>/dev/null || true
        if "$PY" -c "import bitsandbytes" 2>/dev/null; then
            log "bitsandbytes 0.41.3 OK"
            STATUS[bnb]="ok (0.41.3 pinned)"
        else
            warn "bitsandbytes broken — bnb_nf4/bnb_int8 will be skipped in run_quant_comparison.py"
            STATUS[bnb]="SKIP"
        fi
    fi
else
    warn "bitsandbytes install failed — bnb_nf4/bnb_int8 will be skipped"
    STATUS[bnb]="SKIP"
fi

# ── 6. autoawq (optional) ────────────────────────────────────────────────────
# Note: AutoAWQ is deprecated upstream. Last tested with transformers 4.51.3.
# awq_4bit in run_quant_comparison.py may fail with newer transformers versions.
log "Installing autoawq ..."
if "$PIP" install autoawq --quiet 2>/dev/null; then
    if "$PY" -c "import awq" 2>/dev/null; then
        AWQ_VER=$("$PY" -c "import awq; print(getattr(awq, '__version__', '?'))" 2>/dev/null || echo "?")
        log "autoawq $AWQ_VER OK (note: deprecated — awq_4bit may still fail at runtime)"
        STATUS[awq]="ok ($AWQ_VER) — deprecated, runtime failures possible"
    else
        warn "autoawq installed but import fails — awq_4bit will be skipped"
        STATUS[awq]="SKIP"
    fi
else
    warn "autoawq install failed — awq_4bit will be skipped"
    STATUS[awq]="SKIP"
fi

# ── 7. GPTQModel (optional) ───────────────────────────────────────────────────
if [ -d "$GPTQMODEL_DIR" ]; then
    log "Installing GPTQModel from $GPTQMODEL_DIR (no-deps) ..."
    if "$PIP" install -e "$GPTQMODEL_DIR" --no-deps --quiet 2>/dev/null; then
        IMPORT_ERR=$("$PY" -c "import gptqmodel" 2>&1 || true)
        if [ -z "$IMPORT_ERR" ]; then
            log "gptqmodel OK"
            STATUS[gptqmodel]="ok"
        else
            MISSING=$(echo "$IMPORT_ERR" | grep -oP "No module named '\K[^']+" | head -1)
            if [ -n "$MISSING" ]; then
                warn "gptqmodel missing dep: $MISSING — trying to install ..."
                "$PIP" install "$MISSING" --quiet 2>/dev/null || true
                if "$PY" -c "import gptqmodel" 2>/dev/null; then
                    log "gptqmodel OK (after installing $MISSING)"
                    STATUS[gptqmodel]="ok"
                else
                    warn "gptqmodel still broken — run: $PIP install -e $GPTQMODEL_DIR"
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
    warn "GPTQModel repo not found at $GPTQMODEL_DIR — skipping"
    warn "Clone it with: git clone https://github.com/ModelCloud/GPTQModel ~/gptq/GPTQModel"
    STATUS[gptqmodel]="SKIP (repo not found)"
fi

# ── summary ───────────────────────────────────────────────────────────────────
echo ""
log "══════════════════════════════════════════════════════════"
log " Setup complete"
log "══════════════════════════════════════════════════════════"
echo ""
echo "  torch        : ${STATUS[torch]}"
echo "  core deps    : ${STATUS[core]}"
echo "  datautils    : ${STATUS[datautils]}"
echo "  bitsandbytes : ${STATUS[bnb]}"
echo "  autoawq      : ${STATUS[awq]}"
echo "  gptqmodel    : ${STATUS[gptqmodel]}"
echo ""
echo "To activate:"
echo "  module load CUDA/12.1.1"
echo "  source $VENV/bin/activate"
echo "  export HF_HOME=\$SHARED_SCRATCH/hf_cache"
echo "  export HF_DATASETS_CACHE=\$SHARED_SCRATCH/hf_cache/datasets"
echo "  cd ~/gptq/gptq"
echo ""
echo "Scripts and what they need:"
echo ""
echo "  Core experiments (torch + transformers + datasets only):"
echo "    python run_percdamp_sweep.py          # → results_percdamp.csv"
echo "    python run_blocksize_sweep.py         # → results_blocksize.csv"
echo "    python run_mse_experiment.py          # → results_mse.csv"
echo "    python run_mixed_precision_sweep.py   # → results_mixed.csv"
echo ""
echo "  Benchmark comparison (also needs bitsandbytes / autoawq / gptqmodel):"
echo "    python run_quant_comparison.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0"
echo "                                          # → results_comparison.csv"
