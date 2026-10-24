"""Файлы артефактов моделей: контрольная сумма и загрузка sklearn-модели."""

import hashlib
import pickle
from pathlib import Path

import joblib


def sha256_of(path: Path | str) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_artifact(path: Path):
    """.joblib — из тренера; .pkl — артефакт из ноутбука (pickle.dump)."""
    if path.suffix == ".pkl":
        with path.open("rb") as f:
            return pickle.load(f)
    return joblib.load(path)
