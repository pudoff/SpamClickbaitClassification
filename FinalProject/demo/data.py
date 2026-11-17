"""Загрузка, очистка, валидация и групповой сплит текстового датасета.

Протокол воспроизводит ключевые идеи Datasets/scripts (очистка, дедупликация,
групповое разделение), но в упрощённом самодостаточном виде для демо/Colab.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

LABELS = (0, 1, 2)
LABEL_NAMES = {0: "normal", 1: "spam", 2: "clickbait"}
SEED = 42

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_HTML_RE = re.compile(r"<[^>]{1,200}>")
_NUM_RE = re.compile(r"\d+")
_WS_RE = re.compile(r"\s+")
_CYR_RE = re.compile(r"[а-яё]")


def clean_text(text: str) -> str:
    """Нормализация demo-clean-v1 (упрощённый clean-text-v2 из inference-api/app/text.py).

    Нижний регистр до масок (ссылки КАПСОМ тоже маскируются), ё→е,
    URL→'url', числа→'num', HTML-теги удаляются, пробелы схлопываются.
    """
    t = text.lower().replace("ё", "е")
    t = _URL_RE.sub(" url ", t)
    t = _HTML_RE.sub(" ", t)
    t = _NUM_RE.sub(" num ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def template_key(cleaned: str) -> str:
    """Ключ группы близких дубликатов: текст без числовых заполнителей.

    Спам-шаблоны, отличающиеся только суммами/номерами/датами, попадают
    в одну группу и не могут оказаться по разные стороны сплита.
    """
    return _WS_RE.sub(" ", cleaned.replace("num", " ")).strip()


@dataclass
class LoadReport:
    n_lines: int = 0
    n_parsed: int = 0
    skipped: Counter = field(default_factory=Counter)
    n_duplicates_removed: int = 0
    n_conflict_texts_removed: int = 0
    n_short_removed: int = 0

    def as_dict(self) -> dict:
        return {
            "n_lines": self.n_lines,
            "n_parsed": self.n_parsed,
            "skipped": dict(self.skipped),
            "n_duplicates_removed": self.n_duplicates_removed,
            "n_conflict_texts_removed": self.n_conflict_texts_removed,
            "n_short_removed": self.n_short_removed,
        }


def load_dataset(path, max_rows: int | None = None, seed: int = SEED):
    """Читает CSV (text,label), чистит и дедуплицирует; возвращает (texts_raw, texts_clean, y, report).

    Устойчив к «грязным» строкам df.csv: записи через ';', встроенные заголовки,
    нечисловые и недопустимые метки (в файле встречается метка 2000).
    """
    report = LoadReport()
    pairs: dict[str, int] = {}
    conflicts: set[str] = set()
    raw_of: dict[str, str] = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            report.n_lines += 1
            if len(row) == 1 and ";" in row[0]:
                row = row[0].rsplit(";", 1)
            if len(row) != 2:
                report.skipped["bad_column_count"] += 1
                continue
            text, label_raw = row[0], row[1].strip()
            if text.strip().lower() == "text":
                report.skipped["embedded_header"] += 1
                continue
            try:
                label = int(label_raw)
            except ValueError:
                report.skipped["no_integer_label"] += 1
                continue
            if label not in LABELS:
                report.skipped["invalid_label"] += 1
                continue
            cleaned = clean_text(text)
            if len(cleaned.split()) < 3 or not _CYR_RE.search(cleaned):
                report.n_short_removed += 1
                continue
            report.n_parsed += 1
            if cleaned in pairs:
                if pairs[cleaned] != label:
                    conflicts.add(cleaned)
                else:
                    report.n_duplicates_removed += 1
                continue
            pairs[cleaned] = label
            raw_of[cleaned] = text
    for c in conflicts:
        pairs.pop(c, None)
        raw_of.pop(c, None)
    report.n_conflict_texts_removed = len(conflicts)

    items = sorted(pairs.items())  # детерминированный порядок до сэмплирования
    if max_rows is not None and max_rows < len(items):
        rng = np.random.default_rng(seed)
        y_all = np.array([lab for _, lab in items])
        idx = []
        for lab in LABELS:  # стратифицированная подвыборка по классам
            pool = np.flatnonzero(y_all == lab)
            take = max(1, round(max_rows * len(pool) / len(items)))
            idx.extend(rng.choice(pool, size=min(take, len(pool)), replace=False))
        items = [items[i] for i in sorted(idx)]

    texts_clean = [t for t, _ in items]
    y = np.array([lab for _, lab in items], dtype=int)
    texts_raw = [raw_of[t] for t in texts_clean]
    return texts_raw, texts_clean, y, report


def validate_dataset(texts_clean: list[str], y: np.ndarray) -> dict:
    """Проверки качества данных (по образцу практики В06 курса). Возвращает отчёт и список проблем."""
    issues: list[str] = []
    n = len(texts_clean)
    lengths = np.array([len(t.split()) for t in texts_clean])
    class_counts = {LABEL_NAMES[int(k)]: int(v) for k, v in zip(*np.unique(y, return_counts=True))}

    if n == 0:
        issues.append("пустой датасет")
    if set(np.unique(y)) - set(LABELS):
        issues.append("недопустимые метки")
    if len(set(texts_clean)) != n:
        issues.append("остались точные дубликаты после очистки")
    keys = [template_key(t) for t in texts_clean]
    key_labels: dict[str, set] = {}
    for k, lab in zip(keys, y):
        key_labels.setdefault(k, set()).add(int(lab))
    n_conflict_groups = sum(1 for v in key_labels.values() if len(v) > 1)
    if n_conflict_groups:
        issues.append(f"{n_conflict_groups} шаблонных групп со смешанными метками")
    short_share = float((lengths < 4).mean()) if n else 0.0
    if short_share > 0.05:
        issues.append(f"доля очень коротких текстов (<4 слов): {short_share:.1%}")
    minority = min(class_counts.values()) / n if n else 0.0
    if minority < 0.10:
        issues.append(f"сильный дисбаланс классов: миноритарный класс {minority:.1%}")

    return {
        "n_rows": n,
        "class_counts": class_counts,
        "n_template_groups": len(key_labels),
        "n_conflict_groups": n_conflict_groups,
        "words": {
            "mean": round(float(lengths.mean()), 1) if n else 0,
            "median": float(np.median(lengths)) if n else 0,
            "p95": float(np.percentile(lengths, 95)) if n else 0,
            "max": int(lengths.max()) if n else 0,
        },
        "short_share_lt4_words": round(short_share, 4),
        "issues": issues,
    }


def group_split(texts_clean: list[str], y: np.ndarray, seed: int = SEED):
    """Групповой стратифицированный сплит fit/dev/test ≈ 72/8/20 по группам шаблонов.

    Группа — template_key; метка группы — самая частая (при равенстве меньшая).
    Возвращает индексы (fit_idx, dev_idx, test_idx); пересечений групп нет.
    """
    groups: dict[str, list[int]] = {}
    for i, t in enumerate(texts_clean):
        groups.setdefault(template_key(t), []).append(i)
    by_class: dict[int, list[str]] = {lab: [] for lab in LABELS}
    for key, idx in groups.items():
        counts = Counter(int(y[i]) for i in idx)
        lab = min(counts, key=lambda c: (-counts[c], c))
        by_class[lab].append(key)
    rng = np.random.default_rng(seed)
    parts: dict[str, list[int]] = {"fit": [], "dev": [], "test": []}
    for lab in LABELS:
        keys = sorted(by_class[lab])
        rng.shuffle(keys)
        n_test = max(1, round(0.20 * len(keys)))
        n_dev = max(1, round(0.10 * (len(keys) - n_test)))
        for key in keys[:n_test]:
            parts["test"].extend(groups[key])
        for key in keys[n_test : n_test + n_dev]:
            parts["dev"].extend(groups[key])
        for key in keys[n_test + n_dev :]:
            parts["fit"].extend(groups[key])
    return (np.array(sorted(parts["fit"])), np.array(sorted(parts["dev"])), np.array(sorted(parts["test"])))