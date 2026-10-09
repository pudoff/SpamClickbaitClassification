"""FastAPI-приложение: /predict, /healthcheck, /model-info. Swagger UI — /docs."""

import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, RedirectResponse

from app.config import Settings
from app.model import LABELS, LoadedModel, decide, load_model
from app.schemas import HealthResponse, ModelInfo, PredictRequest, PredictResponse, Scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("inference-api")

LABEL_IDS = {name: idx for idx, name in LABELS.items()}


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
            "Модель — TF-IDF (word + char_wb) + LogisticRegression («Модель 2» из `Datasets/model_mlflow.ipynb`). "
            "Низкая уверенность возвращается как `uncertain`, а не как `normal`."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings

    def get_model(request: Request) -> LoadedModel:
        model = request.app.state.model
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"модель не загружена: {request.app.state.load_error}",
            )
        return model

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    @app.post("/predict", response_model=PredictResponse, tags=["inference"])
    def predict(body: PredictRequest, request: Request) -> PredictResponse:
        """Инференс модели: JSON на входе, предсказание на выходе."""
        model = get_model(request)
        scores = model.predict_proba(body.text)
        decision = decide(scores, settings.tau, settings.tau_spam, body.threshold)
        return PredictResponse(
            **decision,
            label_id=LABEL_IDS[decision["label"]] if decision["label"] else None,
            scores=Scores(**scores),
            model_version=model.model_version,
        )

    @app.get(
        "/healthcheck",
        response_model=HealthResponse,
        tags=["service"],
        responses={503: {"model": HealthResponse, "description": "Модель не загружена"}},
    )
    def healthcheck(request: Request):
        """Сервис жив и модель загружена."""
        model = request.app.state.model
        if model is None:
            body = HealthResponse(status="unavailable", model_loaded=False, detail=request.app.state.load_error)
            return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=body.model_dump())
        return HealthResponse(status="ok", model_loaded=True, model_version=model.model_version)

    @app.get("/model-info", response_model=ModelInfo, tags=["service"])
    def model_info(request: Request) -> ModelInfo:
        """Метаданные модели: версия, дата обучения, метрики."""
        model = get_model(request)
        meta = model.meta
        trained_at = meta.get("trained_at")
        return ModelInfo(
            name=meta.get("name", "tfidf-logreg"),
            version=model.version,
            model_version=model.model_version,
            trained_at=datetime.fromisoformat(trained_at) if trained_at else None,
            framework="scikit-learn",
            sklearn_version=meta.get("sklearn_version"),
            artifact=model.path.name,
            artifact_sha256=model.sha256,
            classes={str(i): name for i, name in LABELS.items()},
            metrics=meta.get("metrics"),
            dataset=meta.get("dataset"),
            config=meta.get("config"),
            policy={"tau": settings.tau, "tau_spam": settings.tau_spam},
        )

    return app


app = create_app()
