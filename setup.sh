#!/bin/bash
set -e

# Lambda Stack 22.04 on H100
# CUDA, cuDNN, NCCL, PyTorch are pre-installed system-wide.

python3 -m venv --system-site-packages .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install matplotlib

echo ""
echo "Done. Activate with:  source .venv/bin/activate"
echo "Smoke test:           python ttt_dreamer.py --num-steps 5 --chunk-size 64"
