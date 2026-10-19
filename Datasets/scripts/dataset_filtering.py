"""Проверка текстов до train/test split; исходный CSV не изменяется."""

from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import re

import pandas as pd

CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
CYRILLIC_WORD_RE = re.compile(r"[А-Яа-яЁё]+(?:[-‐‑][А-Яа-яЁё]+)*")
TOKEN_EDGE_RE = re.compile(r"^[\W_]+|[\W_]+$")
TARGET_COLUMN = "is_spam1_or_clickbait2"


@lru_cache(maxsize=65536)
def token_body(token):
    # Пунктуация и эмодзи вокруг слова не мешают считать его русским словом.
    # Цифры и латинские буквы сохраняются: «слово123» и «словоabc» не проходят.
    return TOKEN_EDGE_RE.sub("", token)


@lru_cache(maxsize=65536)
def token_key(token):
    # Пунктуация вокруг слов не мешает обнаружить «да, ДА да! да».
    # Самостоятельные символы тоже сравниваются, но не склеивают соседние слова.
    return (token_body(token) or token).casefold().replace("ё", "е")


def inspect_text(text):
    """Удалить целиком серии из 4+ одинаковых токенов, затем проверить текст.

    Слово — отдельный пробельный токен из русских букв (включая Ё/ё),
    допускаются дефисы внутри и пунктуация по краям. Смешанные ru/en токены,
    числа, одиночные знаки и ссылки словами на кириллице не считаются.
    Три повтора допустимы; серия из четырёх удаляется целиком, не сокращается.
    """
    if not isinstance(text, str):
        text = ""
    tokens = text.split()
    runs, fragments = [], []
    start = 0
    while start < len(tokens):
        end = start + 1
        key = token_key(tokens[start])
        while end < len(tokens) and token_key(tokens[end]) == key:
            end += 1
        if end - start > 3:
            fragments.append(" ".join(tokens[start:end]))
        else:
            current = tokens[start:end]
            # Удаление мусора может соединить два коротких одинаковых ряда.
            # Удаляем и получившийся повтор: повторная очистка не меняет результат.
            if runs and runs[-1][0] == key:
                current = runs.pop()[1] + current
            if len(current) > 3:
                fragments.append(" ".join(current))
            else:
                runs.append((key, current))
        start = end
    kept = [token for _, run in runs for token in run]
    cleaned = " ".join(kept) if fragments else text
    word_count = sum(
        CYRILLIC_WORD_RE.fullmatch(token_body(token)) is not None
        for token in kept
    )
    reasons = []
    if CYRILLIC_RE.search(cleaned) is None:
        reasons.append("no_cyrillic")
    if word_count < 3:
        reasons.append("fewer_than_3_cyrillic_words")
    return cleaned, word_count, fragments, reasons


def filter_dataset(frame, text_column="text", label_column=TARGET_COLUMN):
    """Возвращает новую таблицу, журнал изменений и статистику.

    Журнал содержит и удалённые записи, и исправленные сохранённые тексты.
    source_line — физическая строка CSV (заголовок=1), если загрузчик сохранил
    её в frame.attrs['source_lines']; иначе поле пустое. source_index всегда
    относится к таблице до фильтрации, а не к физической строке CSV.
    """
    cleaned_texts, keep_positions, changes = [], [], []
    counters = Counter()
    source_lines = frame.attrs.get("source_lines")
    if source_lines is not None and len(source_lines) != len(frame):
        raise ValueError("source_lines не соответствует числу строк DataFrame")
    for position, (source_index, text, label) in enumerate(
        zip(frame.index, frame[text_column], frame[label_column])
    ):
        cleaned, word_count, fragments, reasons = inspect_text(text)
        if label not in {0, 1, 2}:
            reasons.append("invalid_label")
        counters.update(reasons)
        if fragments:
            counters["rows_with_removed_repeats"] += 1
            counters["removed_repeat_fragments"] += len(fragments)
        if reasons:
            counters["excluded_rows"] += 1
        else:
            keep_positions.append(position)
            cleaned_texts.append(cleaned)
            if fragments:
                counters["corrected_retained_rows"] += 1
        if reasons or fragments:
            changes.append({
                "source_index": source_index,
                "source_line": source_lines[position] if source_lines is not None else None,
                "label": label,
                "action": "excluded" if reasons else "corrected",
                "reasons": ";".join(reasons),
                "cyrillic_words_after": word_count,
                "text_before": text,
                "text_after": cleaned,
                "removed_fragments": json.dumps(fragments, ensure_ascii=False),
            })
    filtered = frame.iloc[keep_positions].copy()
    filtered[text_column] = cleaned_texts
    filtered.attrs = {}
    if source_lines is not None:
        filtered.attrs["source_lines"] = [source_lines[p] for p in keep_positions]
    filtered = filtered.reset_index(drop=True)
    columns = ["source_index", "source_line", "label", "action", "reasons",
               "cyrillic_words_after", "text_before", "text_after", "removed_fragments"]
    journal = pd.DataFrame(changes, columns=columns)
    stats = {
        "rules": {"min_cyrillic_words": 3, "remove_runs_longer_than": 3,
                  "repeat_action": "remove_fragment_then_filter"},
        "input_rows": len(frame), "retained_rows": len(filtered),
        **{key: int(counters[key]) for key in (
            "excluded_rows", "no_cyrillic", "fewer_than_3_cyrillic_words",
            "invalid_label", "rows_with_removed_repeats", "removed_repeat_fragments",
            "corrected_retained_rows")},
        "input_classes": {str(k): int(v) for k, v in frame[label_column].value_counts().sort_index().items()},
        "retained_classes": {str(k): int(v) for k, v in filtered[label_column].value_counts().sort_index().items()},
        "note": "Причины исключения пересекаются; счётчики нельзя складывать.",
    }
    return filtered, journal, stats


def save_filter_audit(journal, stats, output_dir):
    """Сохранить исправления и исключения для ручной проверки."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    journal.to_csv(output_dir / "changes.csv", index=False, encoding="utf-8-sig")
    journal.loc[journal.action.eq("excluded")].to_csv(
        output_dir / "excluded.csv", index=False, encoding="utf-8-sig")
    (output_dir / "summary.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
