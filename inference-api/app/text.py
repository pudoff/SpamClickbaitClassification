"""Нормализация текста. Версия записывается в метаданные модели (config.normalizer).

Модель обязана получать ту же нормализацию, на которой обучалась, иначе train/serve skew:
- notebook-clean-text-v1 — clean_text, побайтная копия ноутбуков (1_EDA-…, 2_RNN_CNN_LR,
  Datasets/notebooks/model_mlflow.ipynb). Ссылки маскируются до lower(), поэтому набранные
  КАПСОМ (HTTP://…, WWW.…) не маскируются. Не менять: на ней обучен выпущенный ансамбль.
- clean-text-v2 — clean_text_v2: тот же результат, но lower() в начале, ссылки маскируются
  в любом регистре. На ней обучает тренер LR (training/train.py).
"""

import re
from collections.abc import Callable

URL_RE = re.compile(r"https?://\S+|www\.\S+|<link>")
HTML_RE = re.compile(r"<[^>]+>")
MENTION_RE = re.compile(r"[@#]\w+")
DIGIT_RE = re.compile(r"\d+([.,]\d+)?")
SPACE_RE = re.compile(r"\s+")
NON_LETTER_RE = re.compile(r"[^a-zA-Zа-яА-Я\s]")

NORMALIZER_V1 = "notebook-clean-text-v1"
NORMALIZER_V2 = "clean-text-v2"


def clean_text(text: str, fix_yo: bool = True) -> str:
    """Нормализация сырого текста."""
    text = str(text)
    text = HTML_RE.sub(" ", text)
    text = URL_RE.sub(" URL ", text)
    text = MENTION_RE.sub(" ", text)
    text = text.lower()
    if fix_yo:
        text = text.replace("ё", "е")
    text = DIGIT_RE.sub(" NUM ", text)
    text = NON_LETTER_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def clean_text_v2(text: str, fix_yo: bool = True) -> str:
    """Исправленная нормализация: lower() до масок, поэтому ссылки КАПСОМ тоже становятся «url».

    Отличается от clean_text только ссылками не в нижнем регистре; маски те же:
    «url» строчными (как после lower() в v1), «NUM» заглавными.
    """
    text = str(text).lower()
    if fix_yo:
        text = text.replace("ё", "е")
    text = HTML_RE.sub(" ", text)
    text = URL_RE.sub(" url ", text)
    text = MENTION_RE.sub(" ", text)
    text = DIGIT_RE.sub(" NUM ", text)
    text = NON_LETTER_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


NORMALIZERS: dict[str, Callable[[str], str]] = {NORMALIZER_V1: clean_text, NORMALIZER_V2: clean_text_v2}


def get_normalizer(name: str) -> Callable[[str], str]:
    if name not in NORMALIZERS:
        raise ValueError(f"неизвестная нормализация {name!r}, известны: {sorted(NORMALIZERS)}")
    return NORMALIZERS[name]


def normalizer_of(meta: dict) -> str:
    """Нормализация из метаданных модели; у моделей без записи (выпущены до v2) — v1."""
    return (meta.get("config") or {}).get("normalizer", NORMALIZER_V1)
