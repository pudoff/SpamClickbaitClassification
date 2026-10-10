"""Воспроизводимый первичный EDA: python Datasets/scripts/dataset_eda.py.

Артефакты сохраняются в Datasets/outputs/eda; исходный df.csv не изменяется.
"""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from dataset_utils import CLASS_NAMES, TARGET_COLUMN, load_dataset, text_key
from dataset_paths import DATASET_DIR, EDA_OUTPUT_DIR


def length_summary(values):
    return {"mean": float(values.mean()), "min": int(values.min()),
            "p25": float(values.quantile(.25)), "median": float(values.median()),
            "p75": float(values.quantile(.75)), "p95": float(values.quantile(.95)),
            "p99": float(values.quantile(.99)), "max": int(values.max())}


def duplicate_summary(frame, keys):
    pairs = pd.DataFrame({"key": keys, "label": frame[TARGET_COLUMN]})
    groups = pairs.groupby("key", sort=False)["label"].agg(["size", "nunique"])
    conflicts = groups.index[groups["nunique"] > 1]
    conflict_mask = pairs["key"].isin(conflicts)
    return {
        "unique_texts": int(len(groups)),
        "extra_text_duplicates": int(len(frame) - len(groups)),
        "duplicate_text_groups": int((groups["size"] > 1).sum()),
        "conflicting_text_groups": int(len(conflicts)),
        "conflicting_rows": int(conflict_mask.sum()),
        "conflicting_unique_text_label_pairs": int(pairs[conflict_mask].drop_duplicates().shape[0]),
    }, conflict_mask


def main():
    source = DATASET_DIR / "df.csv"
    output = EDA_OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)
    df, issues, parsing = load_dataset(source)
    issues.to_csv(output / "parsing_issues.csv", index=False, encoding="utf-8-sig")
    text = df["text"]
    chars = text.str.len()
    words = text.str.count(r"\S+")
    flags = pd.DataFrame({
        "empty_or_whitespace": text.str.strip().eq(""),
        "no_cyrillic": ~text.str.contains(r"[А-Яа-яЁё]", regex=True),
        "latin_present": text.str.contains(r"[A-Za-z]", regex=True),
        "url_present": text.str.contains(r"https?://|www\.", regex=True, case=False),
        "html_or_entity": text.str.contains(r"<[^>]+>|&(?:[a-zA-Z]+|#\d+);", regex=True),
        "replacement_character": text.str.contains("\ufffd", regex=False),
        "under_10_chars": chars.lt(10),
        "over_2000_chars": chars.gt(2000),
        "over_512_whitespace_words": words.gt(512),
    })
    valid = df[df[TARGET_COLUMN].isin(CLASS_NAMES)].copy()
    exact_stats, exact_conflicts = duplicate_summary(valid, valid["text"])
    keys = valid["text"].map(text_key)
    normalized_stats, normalized_conflicts = duplicate_summary(valid, keys)
    class_rows = []
    for label, group in valid.groupby(TARGET_COLUMN):
        idx = group.index
        row = {"label": int(label), "name": CLASS_NAMES[label], "rows": len(group),
               "share_percent": len(group) / len(valid) * 100,
               "unique_text_label_pairs": len(group.drop_duplicates()),
               "extra_full_duplicates": int(group.duplicated().sum()),
               "conflicting_rows_exact": int(exact_conflicts.loc[idx].sum()),
               "chars": length_summary(chars.loc[idx]),
               "words": length_summary(words.loc[idx]),
               "flags": {k: int(v) for k, v in flags.loc[idx].sum().items()}}
        class_rows.append(row)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    summary = {
        "file": "Datasets/df.csv", "sha256": digest.hexdigest(),
        "file_bytes": source.stat().st_size, "encoding": "UTF-8",
        "parsing": parsing, "shape": list(df.shape),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "deep_memory_bytes": int(df.memory_usage(index=True, deep=True).sum()),
        "missing_cells": {k: int(v) for k, v in df.isna().sum().items()},
        "label_counts_raw": {str(k): int(v) for k, v in df[TARGET_COLUMN].value_counts().sort_index().items()},
        "extra_full_duplicates_raw": int(df.duplicated().sum()),
        "unique_text_label_pairs_raw": int(len(df.drop_duplicates())),
        "valid_rows": len(valid),
        "unique_text_label_pairs_valid": int(len(valid.drop_duplicates())),
        "exact_text": exact_stats, "normalized_text": normalized_stats,
        "chars": length_summary(chars), "words": length_summary(words),
        "quality_flags": {k: int(v) for k, v in flags.sum().items()},
        "classes": class_rows,
        "most_frequent_texts": [
            {"text": key[:600], "count": int(value)}
            for key, value in text.value_counts().head(12).items()
        ],
    }
    # Конфликты остаются видимыми для ручной проверки, без автоматической переметки.
    conflict_pairs = valid.loc[exact_conflicts].drop_duplicates().copy()
    conflict_pairs["text"] = conflict_pairs["text"].str.slice(0, 1000)
    conflict_pairs.head(300).to_csv(output / "label_conflicts_sample.csv", index=False, encoding="utf-8-sig")
    audit = pd.concat([group.sample(n=min(30, len(group)), random_state=42)
                       for _, group in valid.drop_duplicates().groupby(TARGET_COLUMN)])
    audit = audit.copy()
    audit.insert(0, "recovered_row_index", audit.index)
    audit["reviewed_label"] = ""
    audit["translation_ok"] = ""
    audit["comment"] = ""
    audit.to_csv(output / "manual_audit_sample.csv", index=False, encoding="utf-8-sig")
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{k: row[k] for k in (
        "label", "name", "rows", "share_percent", "unique_text_label_pairs", "extra_full_duplicates")}
        for row in class_rows]).to_csv(output / "class_summary.csv", index=False, encoding="utf-8-sig")
    counts = valid[TARGET_COLUMN].value_counts().reindex(CLASS_NAMES, fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    labels = [f"{k} — {name}" for k, name in CLASS_NAMES.items()]
    colors = ["#2ecc71", "#e74c3c", "#f39c12"]
    axes[0].bar(labels, counts.values, color=colors, edgecolor="black")
    for index, value in enumerate(counts.values):
        axes[0].text(index, value, f"{value:,}", ha="center", va="bottom")
    axes[0].set_title("Распределение допустимых классов")
    axes[0].set_ylabel("Количество записей")
    axes[1].pie(counts.values, labels=labels, colors=colors, autopct="%1.2f%%", startangle=90)
    axes[1].set_title(f"Всего {len(valid):,}; исключена метка вне таксономии")
    fig.tight_layout()
    fig.savefig(output / "class_distribution.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for label in CLASS_NAMES:
        subset = valid.index[valid[TARGET_COLUMN].eq(label)]
        axes[0].hist(chars.loc[subset].clip(upper=2000), bins=60, histtype="step",
                     label=CLASS_NAMES[label], density=True)
        axes[1].hist(words.loc[subset].clip(upper=400), bins=60, histtype="step",
                     label=CLASS_NAMES[label], density=True)
    axes[0].set_xlabel("Длина в символах (значения >2000 включены в крайний интервал)")
    axes[1].set_xlabel("Число слов (значения >400 включены в крайний интервал)")
    for ax in axes:
        ax.set_ylabel("Плотность")
        ax.set_yscale("log")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "text_lengths.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
