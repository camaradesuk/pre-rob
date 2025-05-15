# syntax=docker/dockerfile:1

################################################################################
# Stage 0: Assets - Download large custom models not in the Git repository
################################################################################
FROM python:3.10-slim AS assets
# Using python:3.10-slim as it's a common small base with pip

RUN pip install --no-cache-dir --quiet gdown

RUN mkdir /weights
WORKDIR /weights

# Only download files that are too large for the Git repository.
# For 'welfare' model's large .pth.tar file
RUN gdown --id 18YixZQ4otcZWdAMavy5OviR0579kWrCm -O dsc_w0.pth.tar


################################################################################
# Stage 1: Hugging Face Cache - Pre-download Hugging Face models
################################################################################
FROM python:3.10-slim AS hf-cache-builder
# This stage will have Python 3.10

# Install necessary libraries for downloading models
RUN pip install --no-cache-dir \
    "transformers>=4.30,<5" \
    "sentence-transformers>=2.2.0" \
    "torch" 
    # SentenceTransformer might need torch to load models

# Define where Hugging Face libraries should store cached files within this stage
ENV HF_HOME=/hf_cache_temp
ENV TRANSFORMERS_CACHE=${HF_HOME}/hub
ENV SENTENCE_TRANSFORMERS_HOME=${HF_HOME}/sentence_transformers
ENV HF_DATASETS_CACHE=${HF_HOME}/datasets
ENV HF_HUB_CACHE=${HF_HOME}/hub

RUN mkdir -p $TRANSFORMERS_CACHE $SENTENCE_TRANSFORMERS_HOME $HF_DATASETS_CACHE

# Python script to perform the downloads
RUN python - <<'PY'
import os
from transformers import AutoModel, AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer

# Ensure cache dirs are used, though HF_HOME should handle it
transformers_cache_dir = os.environ.get('TRANSFORMERS_CACHE')
sentence_transformers_cache_dir = os.environ.get('SENTENCE_TRANSFORMERS_HOME')

print(f"--- Pre-downloading Hugging Face Models to {os.environ.get('HF_HOME')} ---")

distilbert_model_name = 'distilbert-base-uncased'
sentence_transformer_model_name = 'distilbert-base-nli-stsb-mean-tokens'

print(f"Downloading tokenizer, config, and model for: {distilbert_model_name}")
AutoTokenizer.from_pretrained(distilbert_model_name, cache_dir=transformers_cache_dir)
AutoConfig.from_pretrained(distilbert_model_name, cache_dir=transformers_cache_dir)
AutoModel.from_pretrained(distilbert_model_name, cache_dir=transformers_cache_dir)
print(f"Finished downloading: {distilbert_model_name}")

print(f"Downloading sentence transformer model: {sentence_transformer_model_name}")
SentenceTransformer(sentence_transformer_model_name, cache_folder=sentence_transformers_cache_dir)
print(f"Finished downloading: {sentence_transformer_model_name}")

print("--- Hugging Face Models Pre-download Complete in hf-cache-builder stage ---")
PY

################################################################################
# Stage 2: Base Runtime - Uses the official PyTorch image
################################################################################
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime AS base
# This image has Python 3.10

# Re‑use the already‑installed compiler tool‑chain for native wheels (if any pip installs here need compilation)
RUN printf "[global]\nno-build-isolation = yes\n" > /etc/pip.conf

# Compiler tool‑chain needed by a few older deps (potentially for spacy if not fully wheeled)
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

# --- Environment Variables ---
# For application home and custom models path (relative to rob.py location)
ENV APP_HOME=/pre-rob/rob-app
ENV PTH_DIR=${APP_HOME}/pth 
# This is where rob.py will look for the 'pth' directory

# For Hugging Face cache location in the final image
ENV HF_HOME=/opt/hf_cache
ENV TRANSFORMERS_CACHE=${HF_HOME}/hub
ENV SENTENCE_TRANSFORMERS_HOME=${HF_HOME}/sentence_transformers
ENV HF_DATASETS_CACHE=${HF_HOME}/datasets
ENV HF_HUB_CACHE=${HF_HOME}/hub 
# Explicitly set for transformers

# Enforce offline mode for Hugging Face libraries
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1 
# Good practice, though datasets lib might not be used directly

# Create directories
RUN mkdir -p $APP_HOME $HF_HOME $TRANSFORMERS_CACHE $SENTENCE_TRANSFORMERS_HOME $HF_DATASETS_CACHE $PTH_DIR

# Create a temporary location in the base stage to hold files copied from the assets stage
RUN mkdir -p /tmp_assets_weights

# Copy the downloaded large model(s) from the 'assets' stage's /weights directory
# This makes files from 'assets' available in the 'base' stage at /tmp_assets_weights/
COPY --from=assets /weights/ /tmp_assets_weights/

# Argument to control custom weight inclusion (for dev vs. prod builds)
ARG DOWNLOAD_WEIGHTS=true 
# Default to true for production-ready images

# Now, copy from the temporary location in the base stage to the final PTH_DIR
# This logic runs if DOWNLOAD_WEIGHTS is true, using the files brought in by COPY --from=assets
RUN if [ "$DOWNLOAD_WEIGHTS" = "true" ]; then \
        echo "Copying large model(s) from /tmp_assets_weights to $PTH_DIR"; \
        # Ensure PTH_DIR exists (already created above)
        if [ -f /tmp_assets_weights/dsc_w0.pth.tar ]; then \
            cp /tmp_assets_weights/dsc_w0.pth.tar ${PTH_DIR}/dsc_w0.pth.tar; \
            echo "Copied dsc_w0.pth.tar to ${PTH_DIR}"; \
        else \
            echo "Warning: /tmp_assets_weights/dsc_w0.pth.tar not found after copying from assets stage."; \
        fi; \
    else \
        echo "DEV IMAGE: DOWNLOAD_WEIGHTS=false. Skipping copy of large model(s) from /tmp_assets_weights."; \
        echo "Expecting volume mount for these files in $PTH_DIR if needed."; \
    fi \
    && rm -rf /tmp_assets_weights
    # Clean up temporary directory

# Copy pre-cached Hugging Face models from 'hf-cache-builder' stage
COPY --from=hf-cache-builder /hf_cache_temp/ ${HF_HOME}/
RUN echo "Hugging Face cache copied to $HF_HOME" 
# Removed ls -R for cleaner logs, can be added for debugging


################################################################################
# Stage 3: Build Conda Environment
################################################################################
FROM base AS build-env
# This stage inherits ENV VARS from 'base'

WORKDIR ${APP_HOME} 
# Set working dir for rob-app files

# Copy environment and setup scripts first for better layer caching
COPY rob-app/environment.yml rob-app/setup.sh ./

# Run setup (which installs conda env, pip packages, and spacy model)
RUN chmod +x ./setup.sh \
 && ./setup.sh
# ⬅ creates env “rob”

# Set PATH to include the Conda environment's bin directory for runtime
ENV PATH="/opt/conda/envs/rob/bin:${PATH}"

################################################################################
# Stage 4: Final Image
################################################################################
FROM build-env AS final
# This stage inherits ENV VARS from 'base' (like HF_HOME, PTH_DIR, APP_HOME)

WORKDIR ${APP_HOME}

# Copy all application code from the build context.
# This includes rob.py, rob_fn.py, model.py, and the 'pth' directory
# (containing .json, .Field, and smaller .pth.tar files from your Git repo).
# This will place your local rob-app/pth/* into ${APP_HOME}/pth/
# This will also overwrite any files like setup.sh, environment.yml if they are in rob-app/
COPY rob-app/ ${APP_HOME}/

# Ensure the PTH_DIR (e.g., ${APP_HOME}/pth) has the correct files.
# The dsc_w0.pth.tar should have been copied from the 'assets' stage via 'base'.
# dsc_w0.json and other files in 'pth/' are copied from the local build context by the COPY command above.


# Optional verification steps (uncomment to debug during build)
# RUN echo "Contents of PTH_DIR (${PTH_DIR}) after all copies:" && ls -l ${PTH_DIR}
# RUN conda run -n rob python -c "import spacy; spacy.load('en_core_web_sm'); print('SpaCy en_core_web_sm loaded successfully in final stage.')"
# RUN conda run -n rob python -c "from transformers import AutoConfig; import os; print(f'Attempting to load DistilBERT config from: {os.environ.get(\"TRANSFORMERS_CACHE\")}'); AutoConfig.from_pretrained('distilbert-base-uncased', cache_dir=os.environ.get('TRANSFORMERS_CACHE')); print('DistilBERT config loaded successfully from cache in final stage.')"

ENTRYPOINT ["python", "rob.py"]
# Example CMD for testing (can be overridden with docker run or docker-compose)
# CMD ["-i", "/input/input.csv", "-o", "/output/output.csv", "-s", "0"]

