#!/usr/bin/env bash
set -e
# Script assumes it's run from the directory where environment.yml is located (e.g., /pre-rob/rob-app)
# cd /pre-rob/rob-app # This might be needed if script is called from elsewhere, but Dockerfile WORKDIR handles it.

echo "🟡  Creating conda environment 'rob' from environment.yml..."
conda env create -f environment.yml

echo " activating conda environment 'rob'..."
# Correct way to activate conda in a script
source /opt/conda/etc/profile.d/conda.sh
conda activate rob

echo "🐍 Conda environment 'rob' activated. Python version:"
python --version
echo "🐍 pip version:"
pip --version

echo "📦  Installing runtime packages specified in setup.sh..."
# 1️⃣ ensure the compatible NumPy version is in place (already handled by conda env create if specified in environment.yml)
# If not, it would be:
# pip install --no-cache-dir "numpy==1.24.4" 

# 2️⃣ Install other pip packages (many are already from environment.yml's pip section)
# The environment.yml should be the primary source for these.
# This pip install section in setup.sh can be for packages not suitable for environment.yml
# or to ensure specific versions if needed post-conda setup.
# For example, if torchtext needed a very specific build.
# However, your environment.yml already lists:
#    - torchtext==0.17.2
#    - spacy==2.3.9
#    - transformers>=4.30,<5
#    - sentence-transformers>=2.2.0
#    - dill==0.3.8
#    - debugpy
# So, these don't need to be re-installed here unless there's a specific reason.
# The pandas and numpy are also in environment.yml.
# The gdown is for the Dockerfile's assets stage, not runtime.

# pip install --no-cache-dir \
#     pandas==1.5.3 \ # From environment.yml
#     spacy==2.3.9 \ # From environment.yml
#     torchtext==0.17.2 \ # From environment.yml
#     "transformers>=4.30,<5" \ # From environment.yml
#     "sentence-transformers>=2.2.0" \ # From environment.yml
#     "dill==0.3.8" \ # From environment.yml
#     debugpy # From environment.yml

echo "📦  Installing spaCy model en_core_web_sm v2.3.1 directly via pip..."
# This ensures the exact compatible model for spacy 2.3.9 is installed into the conda env's site-packages.
pip install --no-cache-dir https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz

# The original spacy download command is now replaced by the direct pip install above.
# python -m spacy download en_core_web_sm --direct --no-warn

echo "✅  Setup complete."
