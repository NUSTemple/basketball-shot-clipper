# Labeling UI only - the detect/clip/calibrate CLI tools need ultralytics/opencv
# and 4K source video that don't belong in a container. See docker-compose.yml
# for how clips/dataset volumes are wired up. ffmpeg itself is still needed
# here (not just in native mode) for the Library tab's thumbnails.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==2.1.4

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --without ml,dev --no-interaction --no-ansi

COPY README.md ./
COPY src ./src
RUN poetry install --only-root --no-interaction --no-ansi

ENV SHOT_CLIPPER_CLIPS_DIR=/clips
EXPOSE 5050
CMD ["shot-clipper-label-ui", "--host", "0.0.0.0", "--port", "5050"]
