"""Независимые fit/dev/test с группировкой по нормализованному тексту."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_PROTOCOL = "grouped-fit-dev-test-macro-v1"


def make_data_splits(
    frame,
    normalizer,
    *,
    random_state=42,
    test_size=0.2,
    dev_size=0.1,
    text_column="text",
    label_column="is_spam1_or_clickbait2",
):
    """20% групп в test, 10% оставшихся групп в dev, остальные в fit.

    Все варианты одного нормализованного текста остаются в одной части.
    Стратификация групп использует наиболее частую метку группы (при равенстве
    меньшую метку). Смешанные метки сохраняются и учитываются в диагностике.
    Пропорции строк могут немного отличаться от целевых 72/8/20 из-за групп.
    normalizer — фиксированная функция без обучения на данных.
    """
    if not 0 < test_size < 1 or not 0 < dev_size < 1:
        raise ValueError("test_size и dev_size должны быть между 0 и 1")
    labels = frame[label_column].astype(int).to_numpy()
    if not set(labels).issubset({0, 1, 2}):
        raise ValueError("Перед разделением исключите недопустимые метки")
    keys = frame[text_column].map(normalizer).str.casefold()
    keys = keys.str.replace(r"\s+", " ", regex=True).str.strip()
    grouped = pd.DataFrame({"group": keys.to_numpy(), "label": labels})
    counts = grouped.groupby(["group", "label"], sort=True).size().reset_index(name="count")
    representatives = (
        counts.sort_values(["group", "count", "label"], ascending=[True, False, True])
        .drop_duplicates("group")
        .set_index("group")["label"]
    )
    pool_groups, test_groups = train_test_split(
        representatives.index.to_numpy(),
        test_size=test_size,
        random_state=random_state,
        stratify=representatives.to_numpy(),
    )
    fit_groups, dev_groups = train_test_split(
        pool_groups, test_size=dev_size, random_state=random_state, stratify=representatives.loc[pool_groups].to_numpy()
    )
    indices = {
        name: np.flatnonzero(keys.isin(groups).to_numpy())
        for name, groups in (("fit", fit_groups), ("dev", dev_groups), ("test", test_groups))
    }
    memberships = np.full(len(frame), "", dtype="<U4")
    for name, idx in indices.items():
        memberships[idx] = name
        if set(labels[idx]) != {0, 1, 2}:
            raise ValueError(f"В {name} отсутствует один из классов: увеличьте выборку")
    if not np.all(memberships != ""):
        raise AssertionError("Не все записи распределены")
    group_sets = {name: set(keys.iloc[idx]) for name, idx in indices.items()}
    overlaps = {
        f"{a}_{b}": len(group_sets[a] & group_sets[b]) for a, b in (("fit", "dev"), ("fit", "test"), ("dev", "test"))
    }
    if any(overlaps.values()):
        raise AssertionError("Обнаружено пересечение нормализованных текстов")
    fingerprint = hashlib.sha256()
    fingerprint.update(pd.util.hash_pandas_object(frame[[text_column, label_column]], index=False).to_numpy().tobytes())
    fingerprint.update(pd.util.hash_pandas_object(keys, index=False).to_numpy().tobytes())
    fingerprint.update(memberships.tobytes())
    fingerprint.update(SPLIT_PROTOCOL.encode())
    summary = {
        "protocol": SPLIT_PROTOCOL,
        "split_id": fingerprint.hexdigest(),
        "random_state": random_state,
        "test_group_fraction": test_size,
        "dev_fraction_of_remaining_groups": dev_size,
        "rows": len(frame),
        "normalized_groups": len(representatives),
        "groups_with_conflicting_labels": int((grouped.groupby("group").label.nunique() > 1).sum()),
        "overlapping_normalized_groups": overlaps,
        "partitions": {
            name: {
                "rows": len(idx),
                "groups": len(group_sets[name]),
                "class_counts": {str(c): int((labels[idx] == c).sum()) for c in (0, 1, 2)},
            }
            for name, idx in indices.items()
        },
    }
    return {**indices, "keys": keys, "summary": summary}


def save_split_manifest(frame, split, output_dir):
    """Индексы относятся к очищенной таблице перед разделением, не к строкам CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(
        {
            "prepared_row": np.arange(len(frame)),
            "label": frame["is_spam1_or_clickbait2"].to_numpy(),
            "normalized_group_hash": split["keys"]
            .map(lambda key: hashlib.sha256(key.encode("utf-8")).hexdigest())
            .to_numpy(),
            "partition": "",
        }
    )
    for name in ("fit", "dev", "test"):
        manifest.loc[split[name], "partition"] = name
    manifest.to_csv(output_dir / "manifest.csv", index=False, encoding="utf-8-sig")
    (output_dir / "summary.json").write_text(
        json.dumps(split["summary"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
