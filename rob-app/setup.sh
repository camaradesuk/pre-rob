set -e
echo "Setting conda environment.."
cd rob-pre/rob-app
conda env create --file env_rob.yaml
conda activate rob

echo "Downloading spacy module.."
python -m spacy download en_core_web_sm

echo "====================================================="
echo "Loading pre-trained weights.."
gdown "https://drive.google.com/uc?id=18YixZQ4otcZWdAMavy5OviR0579kWrCm" -O pth/dsc_w0.pth.tar
echo "Finished."