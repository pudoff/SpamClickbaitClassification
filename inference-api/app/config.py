"""Настройки сервиса из переменных окружения (см. .env.example в корне репозитория)."""

import os
from dataclasses import dataclass
from pathlib import Path


def _unit_float(name: str, default: float) -> float:
    value = float(os.getenv(name, default))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name}={value}: порог должен быть в диапазоне [0, 1]")
    return value


@dataclass(frozen=True)
class Settings:
    model_path: Path
    meta_path: Path
    # Политика решения из 02-final-concept.md §6 и 04-service-contract.openapi.yaml (ServiceInfo.policy)
    tau: float
    tau_spam: float

    @classmethod
    def from_env(cls) -> "Settings":
        model_path = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
        meta_path = Path(os.getenv("MODEL_META_PATH", model_path.with_name("model_meta.json")))
        return cls(
            model_path=model_path,
            meta_path=meta_path,
            tau=_unit_float("TAU", 0.55),
            tau_spam=_unit_float("TAU_SPAM", 0.65),
        )
