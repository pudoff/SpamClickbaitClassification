"""Чтение df.csv и подключение Datasets/scripts: общее для train.py и export_ensemble.py."""

import sys
from collections import Counter
from pathlib import Path

from app.labels import LABELS

VALID_LABELS = set(LABELS)


def load_fixed_csv(path: Path) -> tuple[list[str], list[int], Counter]:
    """Построчный разбор df.csv, как в ноутбуке: метка — после последней ',' или ';'.

    Строки без разделителя или с нечисловой меткой пропускаются — в т.ч. маркеры
    merge-конфликта (<<<<<<< / ======= / >>>>>>>), оставшиеся в df.csv.
    """
    texts, labels, bad = [], [], Counter()
    with path.open(encoding="utf-8-sig") as f:
        next(f)  # заголовок
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            pos = max(line.rfind(","), line.rfind(";"))
            if pos == -1:
                bad["нет разделителя"] += 1
                continue
            try:
                label = int(line[pos + 1 :].strip().strip('"'))
            except ValueError:
                bad["метка не число"] += 1
                continue
            if label not in VALID_LABELS:
                bad["метка вне {0,1,2}"] += 1
                continue
            texts.append(line[:pos].strip().strip('"'))
            labels.append(label)
    return texts, labels, bad


def deduplicate(texts: list[str], labels: list[int]) -> tuple[list[str], list[int], dict]:
    """Удаляет полные дубли и тексты с противоречивыми метками (ячейка очистки ноутбука)."""
    pairs = list(dict.fromkeys(zip(texts, labels, strict=True)))
    label_sets: dict[str, set[int]] = {}
    for text, label in pairs:
        label_sets.setdefault(text, set()).add(label)
    conflicting = {t for t, s in label_sets.items() if len(s) > 1}
    kept = [(t, lb) for t, lb in pairs if t not in conflicting]
    stats = {"duplicates_removed": len(texts) - len(pairs), "conflicting_texts_removed": len(conflicting)}
    return [t for t, _ in kept], [lb for _, lb in kept], stats


def import_dataset_scripts(data_path: Path):
    """Один раз добавляет Datasets/scripts (рядом с df.csv) в sys.path; возвращает filter_dataset, make_data_splits."""
    scripts_dir = str(data_path.parent / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from dataset_filtering import filter_dataset
    from dataset_splitting import make_data_splits

    return filter_dataset, make_data_splits
