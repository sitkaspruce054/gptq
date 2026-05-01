# Reproducing the experiments

This document describes how to reproduce the five experiments referenced in the
report: the OPT-125M paper reproduction, the percentage-dampening sweep, the
block-size sweep, the min--max vs MSE-optimal scaling comparison, and the
mixed-precision sensitivity experiment.

All five drivers wrap the upstream `opt.py` entry point and write one CSV
each.

## Hardware

- One NVIDIA GPU.

## Software

- Python 3.10 or newer (3.11.5 was used).
- CUDA 12.1.
- Python packages (installed by `setup.sh`):
  `torch` (cu121 wheel), `transformers>=4.35,<5.0`, `datasets`, `accelerate`,
  `numpy`.

## Setup

```bash
# On the Rice NOTS cluster, load the environment modules first:
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1

# Create the venv and install everything (writes to $SHARED_SCRATCH/gptq-env-311
# to avoid the home-quota limit):
bash setup.sh

# Activate:
source $SHARED_SCRATCH/gptq-env-311/bin/activate
```

For NOTS, set `SHARED_SCRATCH` to a writable directory with at
least 10 GB free (used for the venv, pip cache, and HuggingFace model/dataset
cache):

```bash
export SHARED_SCRATCH=/tmp/gptq-scratch
mkdir -p "$SHARED_SCRATCH"
bash setup.sh
source "$SHARED_SCRATCH/gptq-env-311/bin/activate"
```

## Running all experiments

```bash
bash run_all.sh
```

This runs the five experiments sequentially (they share GPU memory, so they
cannot run in parallel) and writes one CSV per experiment in the current
directory.

The five drivers and their outputs:

- `run_opt125m_results.py` writes `results_opt125m.csv` (paper reproduction).
- `run_percdamp_sweep.py` writes `results_percdamp.csv` (Hessian dampening sweep).
- `run_blocksize_sweep.py` writes `results_blocksize.csv` (block-size sweep).
- `run_mse_experiment.py` writes `results_mse.csv` (min--max vs MSE-optimal scaling).
- `run_mixed_precision_sweep.py` writes `results_mixed.csv` (sensitivity-guided mixed 3/4-bit).

## Running a single experiment

Each driver can be invoked directly. They all accept `--model` (default
`facebook/opt-125m`) and `--output` (default per the list above):

```bash
python run_opt125m_results.py
python run_percdamp_sweep.py
python run_blocksize_sweep.py
python run_mse_experiment.py
python run_mixed_precision_sweep.py
```

## Notes

- `setup.sh` also patches `datautils.py` checks for `trust_remote_code=True`
  (PTB) and `verification_mode='no_checks'` plus the `'en'` config (C4). These
  are needed for the current `datasets` library to load the eval splits used by the paper.
- The HuggingFace cache lives at `$HF_HOME` (default `$SHARED_SCRATCH/hf_cache`)
  and grows to roughly 5 GB after the first run.
