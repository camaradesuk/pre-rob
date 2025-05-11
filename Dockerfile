# ────────────────────────────────────────────────────────────────
# Stage 0  ‑‑  Build the weight file once, in isolation
# ────────────────────────────────────────────────────────────────
FROM python:3.8-slim AS assets

# gdown is tiny; install it only in this throw‑away stage
RUN pip install --no-cache-dir --quiet gdown

# Download to /weights/dsc_w0.pth.tar  (≈700 MB, cached forever after 1st build)
RUN mkdir /weights \
 && gdown --id 18YixZQ4otcZWdAMavy5OviR0579kWrCm \
          -O /weights/dsc_w0.pth.tar

# ────────────────────────────────────────────────────────────────
# Stage 1  ‑‑  Base runtime (PyTorch + CUDA)
# ────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime AS base
ARG DOWNLOAD_WEIGHTS=true           # ‹‑‑ switch prod/dev
ENV PTH_DIR=/pre-rob/pth
WORKDIR /pre-rob

# Copy weight file from the assets stage *into a temporary location*
COPY --from=assets /weights /tmp/weights

# Decide at build time whether to keep the file
RUN if [ "$DOWNLOAD_WEIGHTS" = "true" ]; then \
        echo "→ Keeping weights inside image"; \
        mkdir -p "$PTH_DIR" \
     && mv /tmp/weights/dsc_w0.pth.tar "$PTH_DIR/"; \
    else \
        echo "→ Stripping weights for dev image"; \
    fi \
 && rm -rf /tmp/weights               # always clean the temp dir

# ────────────────────────────────────────────────────────────────
# Stage 2  ‑‑  Build Conda environment
#            (this layer changes most often, but weight layer
#             is already cached so no 700 MB hit)
# ────────────────────────────────────────────────────────────────
FROM base AS build-env

COPY rob-app/environment.yml rob-app/setup.sh /pre-rob/rob-app/
RUN chmod +x /pre-rob/rob-app/setup.sh \
 && /pre-rob/rob-app/setup.sh

ENV PATH="/opt/conda/envs/rob/bin:$PATH"

# ────────────────────────────────────────────────────────────────
# Stage 3  ‑‑  Final runtime image
# ────────────────────────────────────────────────────────────────
FROM build-env AS final

# Copy *all* source code (changes frequently, cheapest layer)
COPY . /pre-rob

ENTRYPOINT [ "python", "/pre-rob/rob-app/rob.py" ]
