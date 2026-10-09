# Один образ для сервиса (uvicorn) и тренера (python -m training.train):
# модель обучается и загружается одной и той же версией scikit-learn.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv

COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements-dev.txt

COPY app ./app
COPY training ./training
COPY tests ./tests
COPY pyproject.toml ./

RUN useradd --create-home --uid 1000 appuser && mkdir -p /models && chown appuser /models /srv
USER appuser

ENV MODEL_PATH=/models/model.joblib
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthcheck', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
