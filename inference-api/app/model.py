"""Загрузка обученной модели (joblib/pickle) и политика решения (шаг 2)."""

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np

from app.text import clean_text

logger = logging.getLogger(__name__)

# Кодировка меток датасета: Datasets/df.csv, поле is_spam1_or_clickbait2
LABELS: dict[int, str] = {0: "normal", 1: "spam", 2: "clickbait"}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_artifact(path: Path):
    """.joblib — из тренера; .pkl — артефакт из ноутбука (pickle.dump)."""
    if path.suffix == ".pkl":
        with path.open("rb") as f:
            return pickle.load(f)
    return joblib.load(path)


@dataclass
class LoadedModel:
    pipeline: object
    meta: dict
    path: Path
    sha256: str
    classes: list[int] = field(default_factory=list)

    @property
    def version(self) -> str:
        return str(self.meta.get("version", "0"))

    @property
    def model_version(self) -> str:
        return f"{self.meta.get('name', 'tfidf-logreg')}@{self.version}"

    def predict_proba(self, text: str) -> dict[str, float]:
        proba = self.pipeline.predict_proba([clean_text(text)])[0]
        return {LABELS[int(c)]: float(p) for c, p in zip(self.classes, proba, strict=True)}


def load_model(model_path: Path, meta_path: Path) -> LoadedModel:
    pipeline = load_artifact(model_path)
    classes = [int(c) for c in np.asarray(pipeline.classes_)]
    if sorted(classes) != sorted(LABELS):
        raise ValueError(f"модель знает классы {classes}, ожидались {sorted(LABELS)}")

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        # Артефакт из ноутбука без метаданных: известны только файл и дата его изменения.
        logger.warning("нет %s — метаданные модели будут неполными", meta_path)
        mtime = datetime.fromtimestamp(model_path.stat().st_mtime, tz=UTC)
        meta = {"name": "tfidf-logreg", "version": "0", "trained_at": mtime.isoformat()}

    model = LoadedModel(pipeline=pipeline, meta=meta, path=model_path, sha256=sha256_of(model_path), classes=classes)
    logger.info("модель %s загружена из %s", model.model_version, model_path)
    return model


def decide(scores: dict[str, float], tau: float, tau_spam: float, threshold: float | None) -> dict:
    """Политика решения 02-final-concept.md §6: низкая уверенность никогда не превращается в normal."""
    label, confidence = max(scores.items(), key=lambda kv: kv[1])
    if threshold is None:
        threshold = tau_spam if label == "spam" else tau
    result = {"confidence": confidence, "threshold": threshold}
    if confidence >= threshold:
        return {**result, "status": "classified", "label": label, "reason": None}
    return {**result, "status": "uncertain", "label": None, "reason": "low_confidence"}
