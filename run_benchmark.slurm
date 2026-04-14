#!/bin/bash
#SBATCH --job-name=quant-bench
#SBATCH --partition=scavenge
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

MODEL="${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
OUTPUT="${OUTPUT:-results_comparison.csv}"

module load CUDA/12.1.1
source "$SHARED_SCRATCH/gptq-env-311/bin/activate"

export HF_HOME="$SHARED_SCRATCH/hf_cache"
export HF_DATASETS_CACHE="$SHARED_SCRATCH/hf_cache/datasets"

cd "$HOME/gptq/gptq"

echo "=== Job $SLURM_JOB_ID | $(date) ==="
echo "Node: $SLURMD_NODENAME"
echo "Model: $MODEL"
echo ""

python run_quant_comparison.py --model "$MODEL" --output "$OUTPUT"

echo ""
echo "=== Done | $(date) ==="
