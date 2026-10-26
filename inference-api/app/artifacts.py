"""Файлы артефактов моделей: контрольная сумма и загрузка sklearn-модели."""

import hashlib
from pathlib import Path

import joblib


def sha256_of(path: Path | str) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_artifact(path: Path):
    """Модель sklearn: .joblib из тренера или .pkl из ноутбука (pickle.dump) — joblib.load читает оба формата."""
    return joblib.load(path)
