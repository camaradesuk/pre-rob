# syntax=docker/dockerfile:1

############################
# Stage 0 — download weights
############################
FROM python:3.8-slim AS assets

RUN pip install --no-cache-dir --quiet gdown
RUN mkdir /weights \
 && gdown --id 18YixZQ4otcZWdAMavy5OviR0579kWrCm \
          -O /weights/dsc_w0.pth.tar


############################
# Stage 1 — base runtime
############################
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime AS base
# Python 3.10

# Re‑use the already‑installed compiler tool‑chain for native wheels
RUN printf "[global]\nno-build-isolation = yes\n" > /etc/pip.conf

# compiler tool‑chain needed by a few older deps
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

ARG DOWNLOAD_WEIGHTS=true
ENV PTH_DIR=/pre-rob/pth
WORKDIR /pre-rob

COPY --from=assets /weights /tmp/weights
RUN if [ "$DOWNLOAD_WEIGHTS" = "true" ]; then \
        echo "→ keeping weights inside the image"; \
        mkdir -p "$PTH_DIR" && mv /tmp/weights/dsc_w0.pth.tar "$PTH_DIR/"; \
    else \
        echo "→ dev image – skipping weight copy"; \
    fi && rm -rf /tmp/weights


############################
# Stage 2 — build Conda env
############################
FROM base AS build-env

COPY rob-app/environment.yml rob-app/setup.sh /pre-rob/rob-app/
RUN chmod +x /pre-rob/rob-app/setup.sh \
 && /pre-rob/rob-app/setup.sh
# ⬅ creates env “rob”

ENV PATH="/opt/conda/envs/rob/bin:${PATH}"


############################
# Stage 3 — final image
############################
FROM build-env AS final
COPY . /pre-rob
ENTRYPOINT ["python", "/pre-rob/rob-app/rob.py"]
