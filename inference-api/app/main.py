"""Классификация текста и аудио, распознавание речи. Swagger UI — /docs."""

import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from threading import Lock
from time import perf_counter
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import FastAPI, Form, Header, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import JSONResponse, RedirectResponse

from app.audio import load_stt, transcribe
from app.config import Settings
from app.labels import LABELS
from app.logs import configure_logging
from app.model import LoadedModel, decide, load_model
from app.schemas import (
    AudioClassificationResponse,
    AudioModels,
    HealthResponse,
    ModelInfo,
    PredictRequest,
    PredictResponse,
    Scores,
    TranscriptionResponse,
)

configure_logging()
logger = logging.getLogger("inference-api")
AUDIO_ERRORS = {
    413: {"description": "Файл больше 20 МиБ"},
    415: {"description": "Неподдерживаемый формат или кодек"},
    422: {"description": "Файл не декодируется, длиннее 300 с или запрос не прошёл валидацию"},
    503: {"description": "Модель недоступна или распознавание занято"},
}


def get_model(request: Request) -> LoadedModel:
    """Загруженная модель или 503.

    Вызывается в теле эндпоинта, а не через Depends: зависимости FastAPI выполняются до валидации тела,
    и некорректный запрос без модели получил бы 503 вместо 422.
    """
    model = request.app.state.model
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"модель не загружена: {request.app.state.load_error}",
        )
    return model


def build_model_info(model: LoadedModel, settings: Settings) -> ModelInfo:
    meta = model.meta
    return ModelInfo(
        name=model.name,
        version=model.version,
        model_version=model.model_version,
        trained_at=model.trained_at,
        framework=meta.get("framework", "scikit-learn"),
        sklearn_version=meta.get("sklearn_version"),
        artifact=model.path.name,
        artifact_sha256=model.sha256,
        classes={str(i): name for i, name in LABELS.items()},
        metrics=meta.get("metrics"),
        dataset=meta.get("dataset"),
        config=meta.get("config"),
        policy={"tau": settings.tau, "tau_spam": settings.tau_spam},
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    def classify(body: PredictRequest, model: LoadedModel) -> PredictResponse:
        scores = model.predict_proba(body.text)
        decision = decide(scores, settings.tau, settings.tau_spam, body.threshold)
        return PredictResponse(**asdict(decision), scores=Scores(**scores), model_version=model.model_version)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = None
        app.state.load_error = None
        try:
            app.state.model = load_model(
                settings.model_path, settings.meta_path, num_threads=settings.torch_num_threads
            )
        except Exception as exc:  # сервис поднимается, но /healthcheck честно отвечает 503
            app.state.load_error = f"{type(exc).__name__}: {exc}"
            logger.error("модель не загружена: %s", app.state.load_error)
        app.state.stt = None
        app.state.stt_lock = Lock()
        if settings.asr_enabled:
            try:
                app.state.stt = load_stt(settings.torch_num_threads)
            except Exception:
                logger.exception("модели распознавания не загружены")
        yield

    app = FastAPI(
        title="Spam & Clickbait Inference API",
        version="0.1.0",
        summary="Распознавание речи и классификация сообщений на spam / clickbait / normal.",
        description=(
            "JSON с текстом принимается в /predict, аудиофайл — в /v1/classifications. "
            "По умолчанию — ансамбль RNN + char-CNN + TF-IDF/LogisticRegression из `Datasets/2_RNN_CNN_LR.ipynb`; "
            "веса выбраны на dev. Загруженная версия и состав доступны через /model-info. "
            "Низкая уверенность возвращается как `uncertain`, а не как `normal`."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    @app.post("/predict", response_model=PredictResponse, tags=["inference"])
    def predict(body: PredictRequest, request: Request) -> PredictResponse:
        """Инференс модели: JSON на входе, предсказание на выходе."""
        return classify(body, get_model(request))

    @app.post("/v1/transcriptions", tags=["audio"], responses=AUDIO_ERRORS)
    def transcriptions(audio: UploadFile, request: Request) -> TranscriptionResponse:
        """Распознать русскую речь из файла без классификации."""
        return transcribe(audio, request)

    @app.post("/v1/classifications", tags=["audio"], responses=AUDIO_ERRORS)
    def classifications(
        audio: UploadFile,
        request: Request,
        response: Response,
        source_kind: Annotated[Literal["own", "forwarded", "record", "file"] | None, Form()] = None,
        x_request_id: Annotated[str | None, Header(max_length=128, pattern=r"^[a-zA-Z0-9._:-]+$")] = None,
    ) -> AudioClassificationResponse:
        """Распознать речь и передать расшифровку существующему классификатору."""
        started = perf_counter()
        model = get_model(request)
        speech = transcribe(audio, request)
        timings = {**speech.timings_ms, "classify": 0}
        prediction = None
        if speech.transcript is not None:
            classify_started = perf_counter()
            prediction = classify(PredictRequest(text=speech.transcript), model)
            timings["classify"] = int((perf_counter() - classify_started) * 1000)
        timings["total"] = int((perf_counter() - started) * 1000)
        request_id = x_request_id or uuid4().hex
        response.headers["X-Request-Id"] = request_id
        return AudioClassificationResponse(
            request_id=request_id,
            status=prediction.status if prediction else "not_classified",
            label=prediction.label if prediction else None,
            reason=prediction.reason if prediction else speech.reason,
            confidence=prediction.confidence if prediction else None,
            scores=prediction.scores if prediction else None,
            detected_language=speech.detected_language,
            audio_duration_s=speech.audio_duration_s,
            speech_duration_s=speech.speech_duration_s,
            transcript=speech.transcript,
            models=AudioModels(
                asr_model=speech.asr_model, lid_model=speech.lid_model, classifier_model=model.model_version
            ),
            timings_ms=timings,
        )

    @app.get(
        "/healthcheck",
        response_model=HealthResponse,
        tags=["service"],
        responses={503: {"model": HealthResponse, "description": "Модель не загружена"}},
    )
    def healthcheck(request: Request):
        """Сервис жив и модель загружена."""
        try:
            model = get_model(request)
        except HTTPException as exc:
            # По контракту 503 приходит с телом HealthResponse, а не с {"detail": ...}
            body = HealthResponse(status="unavailable", model_loaded=False, detail=request.app.state.load_error)
            return JSONResponse(status_code=exc.status_code, content=body.model_dump())
        return HealthResponse(status="ok", model_loaded=True, model_version=model.model_version)

    @app.get("/model-info", response_model=ModelInfo, tags=["service"])
    def model_info(request: Request) -> ModelInfo:
        """Метаданные модели: версия, дата обучения, метрики."""
        return build_model_info(get_model(request), settings)

    return app


app = create_app()
