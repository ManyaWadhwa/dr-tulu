#!/bin/bash
set -e
export TMPDIR=/scratch/mw4141/.tmp
export PIP_CACHE_DIR=/scratch/mw4141/.pip-cache
mkdir -p $TMPDIR $PIP_CACHE_DIR

source /scratch/mw4141/code/dr-tulu/.venv/bin/activate
cd /scratch/mw4141/code/dr-tulu

pip install -r requirements.txt -e agent/ 2>&1 | tee /scratch/mw4141/code/dr-tulu/logs/install.log

echo 'INSTALL COMPLETE' >> /scratch/mw4141/code/dr-tulu/logs/install.log
