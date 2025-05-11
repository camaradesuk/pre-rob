#!/usr/bin/env bash
set -e

echo "Setting conda environment…"
cd /pre-rob/rob-app
conda env create --file environment.yml

source /opt/conda/etc/profile.d/conda.sh
conda activate rob

# (Optional) GPU enable spaCy – remove if you stay CPU‑only
# python - <<'PY'
# import spacy, sys
# spacy.require_gpu()
# print("spaCy GPU availability:", spacy.prefer_gpu())
# PY

# download model (2.x still has the CLI)
python -m spacy download en_core_web_sm

echo "Setup finished successfully."
