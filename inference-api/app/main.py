"""FastAPI-приложение: /predict, /healthcheck, /model-info. Swagger UI — /docs."""

import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, RedirectResponse

from app.config import Settings
from app.labels import LABELS
from app.logs import configure_logging
from app.model import LoadedModel, decide, load_model
from app.schemas import HealthResponse, ModelInfo, PredictRequest, PredictResponse, Scores

configure_logging()
logger = logging.getLogger("inference-api")


def get_model(request: Request) -> LoadedModel:
    """Зависимость эндпоинтов: 503, пока модель не загружена."""
    model = request.app.state.model
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"модель не загружена: {request.app.state.load_error}",
        )
    return model


CurrentModel = Annotated[LoadedModel, Depends(get_model)]


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

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = None
        app.state.load_error = None
        try:
            app.state.model = load_model(settings.model_path, settings.meta_path)
        except Exception as exc:  # сервис поднимается, но /healthcheck честно отвечает 503
            app.state.load_error = f"{type(exc).__name__}: {exc}"
            logger.error("модель не загружена: %s", app.state.load_error)
        yield

    app = FastAPI(
        title="Spam & Clickbait Inference API",
        version="0.1.0",
        summary="Классификация текста сообщения на spam / clickbait / normal.",
        description=(
            "Текстовая часть `inference-api` из `ConceptAndArchitecture/после ревью/.../03-final-architecture.md`. "
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
    def predict(body: PredictRequest, model: CurrentModel) -> PredictResponse:
        """Инференс модели: JSON на входе, предсказание на выходе."""
        scores = model.predict_proba(body.text)
        decision = decide(scores, settings.tau, settings.tau_spam, body.threshold)
        return PredictResponse(**asdict(decision), scores=Scores(**scores), model_version=model.model_version)

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
    def model_info(model: CurrentModel) -> ModelInfo:
        """Метаданные модели: версия, дата обучения, метрики."""
        return build_model_info(model, settings)

    return app


app = create_app()
