"""Нормализация текста — побайтная копия clean_text из Datasets/model_mlflow.ipynb.

Функция обязана совпадать с той, на которой обучалась модель: любое расхождение
даёт train/serve skew. Известная особенность (см. Datasets/test_model.py, тест 4):
URL_RE / MENTION_RE / HTML_RE применяются до lower(), поэтому в КАПСЕ ссылки
маскируются не всегда. Исправлять только вместе с переобучением модели.
"""

import re

URL_RE = re.compile(r"https?://\S+|www\.\S+|<link>")
HTML_RE = re.compile(r"<[^>]+>")
MENTION_RE = re.compile(r"[@#]\w+")
DIGIT_RE = re.compile(r"\d+([.,]\d+)?")
SPACE_RE = re.compile(r"\s+")
NON_LETTER_RE = re.compile(r"[^a-zA-Zа-яА-Я\s]")


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
