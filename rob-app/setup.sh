#!/usr/bin/env bash
set -e
cd /pre-rob/rob-app

echo "🟡  Creating conda environment…"
conda env create -f environment.yml

source /opt/conda/etc/profile.d/conda.sh
conda activate rob

echo "📦  Installing runtime packages…"
# 1️⃣ ensure the compatible NumPy version is in place
pip install --no-cache-dir "numpy==1.24.4"

# 2️⃣ now install everything else
pip install --no-cache-dir \
    pandas==1.5.3 \
    spacy==2.3.9 \
    torchtext==0.17.2 \
    "transformers>=4.30,<5" \
    "sentence-transformers>=2.2.0" \
    "dill==0.3.8" \
    "gdown==5.*" \
    debugpy

# fetch the language model (not on PyPI)
python -m spacy download en_core_web_sm --direct --no-warn

echo "✅  Setup complete."
