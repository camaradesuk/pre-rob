set -e
echo "Setting conda environment..."

# Navigate to the correct directory
cd /pre-rob/rob-app

# Create the conda environment and log errors
conda env create --file env_rob.yaml || {
  echo "Failed to create conda environment"
  exit 1
}

# Activate the environment
echo "Activating conda environment..."
source /opt/conda/etc/profile.d/conda.sh
conda activate rob || {
  echo "Failed to activate conda environment"
  exit 1
}

# Install SpaCy model
echo "Downloading spacy module..."
python -m spacy download en_core_web_sm

echo "Setup finished successfully."
