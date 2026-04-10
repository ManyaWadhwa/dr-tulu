#!/bin/bash
# Run this ONCE on the login node to set up the Python environment.
# Does NOT require a GPU allocation.
set -e

REPO="$(cd "$(dirname "$0")" && pwd)"

echo "=== Setting up DR Tulu environment ==="

# Redirect temp/cache to scratch to avoid /tmp space issues
export TMPDIR=/scratch/$USER/.tmp
export PIP_CACHE_DIR=/scratch/$USER/.pip-cache
mkdir -p $TMPDIR $PIP_CACHE_DIR

# Use uv to create a venv in the repo root
cd "$REPO"
uv venv .venv --python 3.11
source .venv/bin/activate

# Install all dependencies from requirements.txt
uv pip install -r requirements.txt

# Install the agent package in editable mode
cd "$REPO/agent"
uv pip install -e .

# Install rich + typer (needed by scripts)
uv pip install rich typer

echo ""
echo "=== Environment setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Fill in your API keys in $REPO/.env"
echo "     (SERPER_API_KEY, S2_API_KEY, JINA_API_KEY, HF_TOKEN)"
echo ""
echo "  2. Edit run_test_inference.slurm with your partition/account"
echo ""
echo "  3. Submit the SLURM job:"
echo "     sbatch run_test_inference.slurm"
