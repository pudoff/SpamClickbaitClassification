# Один образ для сервиса (uvicorn) и тренера (python -m training.train):
# модель обучается и загружается одной и той же версией scikit-learn.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

FROM base AS poetry

ARG POETRY_VERSION=2.5.1
RUN python -m venv /opt/poetry \
    && /opt/poetry/bin/pip install "poetry==${POETRY_VERSION}" \
    && python -m venv /opt/venv

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

WORKDIR /build
COPY pyproject.toml poetry.lock ./
RUN /opt/poetry/bin/poetry check --lock

FROM poetry AS bot-dependencies
RUN /opt/poetry/bin/poetry sync --only bot --no-root

FROM base AS bot
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /srv
COPY --from=bot-dependencies /opt/venv /opt/venv
COPY inference-api/bot ./bot
RUN useradd --create-home --uid 1000 appuser
USER appuser
CMD ["python", "-m", "bot"]

FROM poetry AS dependencies
RUN /opt/poetry/bin/poetry sync --only main,speech,training,dev --no-root

FROM base AS runtime

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /srv
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser \
    && mkdir -p /models /home/appuser/.cache \
    && chown appuser /models /srv /home/appuser/.cache

COPY --from=dependencies /opt/venv /opt/venv
COPY inference-api/app ./app
COPY inference-api/audio_module ./audio_module
COPY inference-api/training ./training
COPY inference-api/tests ./tests
COPY inference-api/pyproject.toml ./

USER appuser

ENV MODEL_PATH=/models/ensemble/manifest.json
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthcheck', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
