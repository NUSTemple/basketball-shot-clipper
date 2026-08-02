# Includes the `ml` extras (torch/ultralytics/opencv), so the Detect tab's
# actual detection can run here too - not just calibration/thumbnails. Note
# this runs CPU-only: Docker on macOS has no GPU/MPS passthrough, so a long
# 4K video will process noticeably slower here than natively. See
# docker-compose.yml for how clips/dataset/models volumes are wired up.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

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
