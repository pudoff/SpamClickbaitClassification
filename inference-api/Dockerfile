# Один образ для сервиса (uvicorn) и тренера (python -m training.train):
# модель обучается и загружается одной и той же версией scikit-learn.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

FROM base AS dependencies

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
RUN /opt/poetry/bin/poetry check --lock \
    && /opt/poetry/bin/poetry sync --only main,training,dev --no-root

FROM base AS runtime

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /srv
COPY --from=dependencies /opt/venv /opt/venv
COPY inference-api/app ./app
COPY inference-api/training ./training
COPY inference-api/tests ./tests
COPY inference-api/pyproject.toml ./

RUN useradd --create-home --uid 1000 appuser && mkdir -p /models && chown appuser /models /srv
USER appuser

ENV MODEL_PATH=/models/ensemble/manifest.json
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthcheck', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
