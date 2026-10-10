"""Загрузка неоднородного df.csv без изменения исходного файла."""

from collections import Counter
from pathlib import Path
import re

import pandas as pd

from dataset_paths import DATASET_DIR

TARGET_COLUMN = "is_spam1_or_clickbait2"
CLASS_NAMES = {0: "Обычные", 1: "Спам", 2: "Кликбейт"}
LABEL_AT_END = re.compile(r'^(.*)[,;]\s*"?([+-]?\d+)"?\s*$', re.DOTALL)


def resolve_dataset_dir():
    """Путь не зависит от рабочей директории Python/Jupyter."""
    if (DATASET_DIR / "df.csv").is_file():
        return DATASET_DIR
    raise FileNotFoundError(f"df.csv не найден в {DATASET_DIR}")


def load_dataset(filename):
    """Восстановить пары текст/целочисленная метка из физических строк.

    Файл смешивает ',' и ';' и содержит повреждённые фрагменты. Это явно
    заданное восстановление, а не универсальный CSV-парсер: многострочные
    записи не склеиваются и пропуски не исправляются предположениями.
    Недопустимые целочисленные метки остаются в исходном DataFrame для EDA.
    Все непрочитанные строки и метки вне таксономии попадают в журнал.
    """
    records, issues = [], []
    counters = Counter()
    with Path(filename).open(encoding="utf-8-sig", newline="") as stream:
        header = next(stream).rstrip("\r\n")
        if header != f"text,{TARGET_COLUMN}":
            raise ValueError(f"Неожиданный заголовок: {header!r}")
        for line_number, raw in enumerate(stream, start=2):
            counters["physical_data_lines"] += 1
            line = raw.rstrip("\r\n")
            if not line.strip():
                counters["blank_lines"] += 1
                continue
            match = LABEL_AT_END.fullmatch(line)
            if match is None:
                reason = "embedded_header" if line in (
                    "текст,label", "Переводимое название,label"
                ) else "no_integer_label"
                issues.append((line_number, reason, line))
                counters[reason] += 1
                continue
            text, label_string = match.groups()
            text = text.strip()
            # Снимаем только парные внешние CSV-кавычки и декодируем "".
            if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
                text = text[1:-1].replace('""', '"')
            label = int(label_string)
            # Последний разделитель перед целочисленным суффиксом.
            suffix = line[len(match.group(1)):]
            counters["semicolon_records" if suffix.startswith(";") else "comma_records"] += 1
            if label not in CLASS_NAMES:
                issues.append((line_number, "invalid_label", line))
                counters["invalid_label"] += 1
            records.append((text, label))
    frame = pd.DataFrame(records, columns=["text", TARGET_COLUMN])
    issue_frame = pd.DataFrame(issues, columns=["line_number", "reason", "raw_line"])
    stats = dict(counters)
    stats["recovered_records"] = len(frame)
    return frame, issue_frame, stats


def text_key(text):
    """Консервативный ключ для поиска вариантов одного и того же текста."""
    return re.sub(r"\s+", " ", str(text)).strip().casefold().replace("ё", "е")


URL_RE = re.compile(r"https?://\S+|www\.\S+|<link>")
HTML_RE = re.compile(r"<[^>]+>")
MENTION_RE = re.compile(r"[@#]\w+")
DIGIT_RE = re.compile(r"\d+([.,]\d+)?")
SPACE_RE = re.compile(r"\s+")


def clean_text(text, fix_yo=True):
    """Нормализация признаков; URL заменяются до удаления HTML-тегов."""
    text = URL_RE.sub(" URL ", str(text))
    text = HTML_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text).lower()
    if fix_yo:
        text = text.replace("ё", "е")
    text = DIGIT_RE.sub(" num ", text)
    text = re.sub(r"[^a-zA-Zа-яА-Я\s]", " ", text)
    return SPACE_RE.sub(" ", text).strip()
