#!/bin/bash
#SBATCH --job-name=dr-tulu-test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
# >>> UPDATE THESE to match your cluster allocation <<<
#SBATCH --partition=h200_courant
#SBATCH --account=torch_pr_219_courant

set -e

REPO="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$REPO/logs"

echo "=== Job started on $(hostname) at $(date) ==="
nvidia-smi

# Activate the virtual environment
source "$REPO/.venv/bin/activate"

# Cache HuggingFace models on scratch (avoids home quota issues)
export HF_HOME=/scratch/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

# Prevent CUDA fork issues with vLLM
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd "$REPO"

python test_inference.py \
    --query "What are the most recent breakthroughs in large language model alignment research?" \
    --vllm-port 30001 \
    --mcp-port 8000 \
    --model "rl-research/DR-Tulu-8B" \
    --max-model-len 40960 \
    --output "$REPO/logs/result_${SLURM_JOB_ID}.json" \
    --hf-dataset "deepresearchediting/dr-tulu-runs" \
    --verbose

echo "=== Job finished at $(date) ==="
