"""Загрузка обученной модели (joblib/pickle) и политика решения (шаг 2)."""

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.artifacts import load_artifact, sha256_of
from app.labels import LABEL_IDS, LABELS  # app.model.LABELS — прежнее место, импорт сохраняется
from app.text import clean_text

logger = logging.getLogger(__name__)

# Имя по умолчанию для LR без метаданных (артефакт из ноутбука)
DEFAULT_MODEL_NAME = "tfidf-logreg"


@dataclass
class LoadedModel:
    pipeline: object
    meta: dict
    path: Path
    sha256: str
    classes: list[int] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.meta.get("name", DEFAULT_MODEL_NAME)

    @property
    def version(self) -> str:
        return str(self.meta.get("version", "0"))

    @property
    def model_version(self) -> str:
        return f"{self.name}@{self.version}"

    @property
    def trained_at(self) -> datetime | None:
        value = self.meta.get("trained_at")
        return datetime.fromisoformat(value) if value else None

    def predict_proba(self, text: str) -> dict[str, float]:
        proba = self.pipeline.predict_proba([clean_text(text)])[0]
        return {LABELS[int(c)]: float(p) for c, p in zip(self.classes, proba, strict=True)}


def _load_ensemble(path: Path, num_threads: int | None) -> tuple[object, dict]:
    from app.ensemble import load_ensemble  # torch импортируется, только если нужен ансамбль

    pipeline, meta = load_ensemble(path, num_threads=num_threads)
    logger.info("ансамбль: веса %s", pipeline.weights)
    return pipeline, meta


def _load_sklearn(path: Path, meta_path: Path) -> tuple[object, dict]:
    pipeline = load_artifact(path)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        # Артефакт из ноутбука без метаданных: известны только файл и дата его изменения.
        logger.warning("нет %s — метаданные модели будут неполными", meta_path)
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        meta = {"name": DEFAULT_MODEL_NAME, "version": "0", "trained_at": mtime.isoformat()}
    return pipeline, meta


def load_model(model_path: Path, meta_path: Path, *, num_threads: int | None = None) -> LoadedModel:
    """manifest.json — ансамбль; .joblib / .pkl — отдельная LR с model_meta.json."""
    if model_path.suffix == ".json":
        pipeline, meta = _load_ensemble(model_path, num_threads)
    else:
        pipeline, meta = _load_sklearn(model_path, meta_path)
    classes = [int(c) for c in np.asarray(pipeline.classes_)]
    if sorted(classes) != sorted(LABELS):
        raise ValueError(f"модель знает классы {classes}, ожидались {sorted(LABELS)}")
    model = LoadedModel(pipeline=pipeline, meta=meta, path=model_path, sha256=sha256_of(model_path), classes=classes)
    logger.info("модель %s загружена из %s", model.model_version, model_path)
    return model


@dataclass(frozen=True)
class Decision:
    status: str
    label: str | None
    label_id: int | None
    reason: str | None
    confidence: float
    threshold: float


def decide(scores: dict[str, float], tau: float, tau_spam: float, threshold: float | None) -> Decision:
    """Политика решения 02-final-concept.md §6: низкая уверенность никогда не превращается в normal."""
    label, confidence = max(scores.items(), key=lambda kv: kv[1])
    if threshold is None:
        threshold = tau_spam if label == "spam" else tau
    if confidence >= threshold:
        return Decision("classified", label, LABEL_IDS[label], None, confidence, threshold)
    return Decision("uncertain", None, None, "low_confidence", confidence, threshold)
