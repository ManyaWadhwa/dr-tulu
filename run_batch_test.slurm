#!/bin/bash
#SBATCH --job-name=dr-tulu-batch
#SBATCH --output=/scratch/mw4141/code/dr-tulu/logs/batch_%j.out
#SBATCH --error=/scratch/mw4141/code/dr-tulu/logs/batch_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --partition=h200_courant
#SBATCH --account=torch_pr_219_courant

set -e
REPO=/scratch/mw4141/code/dr-tulu
mkdir -p "$REPO/logs"

echo "=== Batch test started on $(hostname) at $(date) ==="
nvidia-smi

source "$REPO/.venv/bin/activate"
export HF_HOME=/scratch/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export VLLM_WORKER_MULTIPROC_METHOD=spawn
cd "$REPO"

python test_inference.py \
    --input-file sample_queries.jsonl \
    --output "$REPO/logs/batch_result_${SLURM_JOB_ID}.jsonl" \
    --no-hf-push \
    --verbose

echo "=== Batch test finished at $(date) ==="
