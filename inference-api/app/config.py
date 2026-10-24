"""Настройки сервиса из переменных окружения (см. .env.example в корне репозитория)."""

import os
from dataclasses import dataclass
from pathlib import Path

# Политика решения из 02-final-concept.md §6 и 04-service-contract.openapi.yaml (ServiceInfo.policy)
DEFAULT_TAU = 0.55
DEFAULT_TAU_SPAM = 0.65
DEFAULT_TORCH_THREADS = 2


def _unit_float(name: str, default: float) -> float:
    value = float(os.getenv(name, default))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name}={value}: порог должен быть в диапазоне [0, 1]")
    return value


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, default))
    if value < 1:
        raise ValueError(f"{name}={value}: нужно целое число не меньше 1")
    return value


@dataclass(frozen=True)
class Settings:
    model_path: Path
    meta_path: Path
    tau: float = DEFAULT_TAU
    tau_spam: float = DEFAULT_TAU_SPAM
    # Потоки torch для ансамбля; на LR не влияет
    torch_num_threads: int = DEFAULT_TORCH_THREADS

    @classmethod
    def from_env(cls) -> "Settings":
        model_path = Path(os.getenv("MODEL_PATH", "models/ensemble/manifest.json"))
        meta_path = Path(os.getenv("MODEL_META_PATH", model_path.with_name("model_meta.json")))
        return cls(
            model_path=model_path,
            meta_path=meta_path,
            tau=_unit_float("TAU", DEFAULT_TAU),
            tau_spam=_unit_float("TAU_SPAM", DEFAULT_TAU_SPAM),
            torch_num_threads=_positive_int("TORCH_NUM_THREADS", DEFAULT_TORCH_THREADS),
        )
