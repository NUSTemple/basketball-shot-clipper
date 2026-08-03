# Includes the `ml` extras (torch/ultralytics/opencv), so the Detect tab's
# actual detection can run here too.
#
# The base is multi-arch on purpose, so this builds on Apple Silicon as well
# as amd64. It does NOT need to be an nvidia/cuda image to use an NVIDIA GPU:
# the linux/amd64 PyPI torch wheel bundles the CUDA runtime through its
# nvidia-* dependencies and only needs the host driver, which the NVIDIA
# container runtime injects. For GPU, add the override file rather than
# changing this line:
#   docker compose -f docker-compose.yml -f docker-compose.nvidia.yml up --build
# To pin a CUDA base anyway:
#   docker build --build-arg BASE_IMAGE=nvidia/cuda:12.3-runtime-ubuntu22.04 .
# See docker-compose.yml for volume setup (clips/dataset/models).
ARG BASE_IMAGE=python:3.12-slim-bookworm
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==2.1.4

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --with ml --without dev --no-interaction --no-ansi

COPY README.md ./
COPY src ./src
RUN poetry install --only-root --no-interaction --no-ansi

ENV SHOT_CLIPPER_CLIPS_DIR=/clips
EXPOSE 5050
CMD ["shot-clipper-label-ui", "--host", "0.0.0.0", "--port", "5050"]
