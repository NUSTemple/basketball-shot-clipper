# Includes the `ml` extras (torch/ultralytics/opencv), so the Detect tab's
# actual detection can run here too. Supports both CPU and NVIDIA GPU:
# - On CPU: runs as before (safe fallback)
# - On NVIDIA GPU: requires docker runtime="nvidia" in docker-compose.yml
#
# For NVIDIA GPU support:
#   docker run --gpus all ...
#   OR in docker-compose.yml: runtime: nvidia
# See docker-compose.yml for volume setup (clips/dataset/models).
FROM nvidia/cuda:12.3-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-distutils \
    ffmpeg \
    libgl1 libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    curl -sSL https://bootstrap.pypa.io/get-pip.py | python3 && \
    pip install --no-cache-dir poetry==2.1.4

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
