"""Обучение «Модели 2» из Datasets/model_mlflow.ipynb и сохранение артефакта для сервиса.

Запуск в Docker (рекомендуется — та же версия sklearn, что у сервиса):
    docker compose --profile train run --rm trainer

Локально (из inference-api/, в .venv проекта):
    python -m training.train --data ../Datasets/df.csv --out models

Результат: <out>/model.joblib и <out>/model_meta.json (версия, дата, метрики, конфигурация).
"""

import argparse
import json
import logging
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from app.model import LABELS, sha256_of
from app.text import clean_text

logger = logging.getLogger("train")

VALID_LABELS = set(LABELS)
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Конфигурации — копия ячейки «эксперименты» ноутбука (BASE_CONFIG, EXP1..EXP5).
BASE_CONFIG = {
    "word_ngram_range": (1, 2),
    "word_min_df": 3,
    "word_max_features": 30000,
    "word_sublinear_tf": True,
    "char_analyzer": "char_wb",
    "char_ngram_range": (3, 5),
    "char_min_df": 3,
    "char_max_features": 30000,
    "char_sublinear_tf": False,
    "union_transformer_weights": None,
    "clf_C": 10.0,
    "clf_class_weight": "balanced",
    "clf_max_iter": 1000,
    "clf_solver": "lbfgs",
}
EXP1_CONFIG = {
    **BASE_CONFIG,
    "word_min_df": 2,
    "word_max_features": 150000,
    "char_min_df": 5,
    "char_max_features": 150000,
}
EXP2_CONFIG = {**EXP1_CONFIG, "word_ngram_range": (1, 3)}
EXP3_CONFIG = {**EXP2_CONFIG, "char_ngram_range": (2, 6), "char_sublinear_tf": True}
EXP4_CONFIG = {**EXP3_CONFIG, "union_transformer_weights": {"word": 1.0, "char": 0.6}}
# EXP5: C=1 и вес кликбейта = 0.7 × balanced; веса считаются по y_train в build_pipeline.
EXP5_CONFIG = {**EXP4_CONFIG, "clf_C": 1.0, "clf_class_weight": "damped_clickbait"}

CONFIGS = {
    "baseline": BASE_CONFIG,
    "exp1": EXP1_CONFIG,
    "exp2": EXP2_CONFIG,
    "exp3": EXP3_CONFIG,
    # Лучший прогон в Datasets/mlflow.db: run_7_best_confusion_matrix -> best_run=run_5_exp4_union_weights
    "exp4": EXP4_CONFIG,
    "exp5": EXP5_CONFIG,
}


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


def build_pipeline(cfg: dict, y_train: np.ndarray) -> Pipeline:
    class_weight = cfg["clf_class_weight"]
    if class_weight == "damped_clickbait":
        classes, counts = np.unique(y_train, return_counts=True)
        balanced = {int(c): round(len(y_train) / (len(classes) * n), 3) for c, n in zip(classes, counts, strict=True)}
        class_weight = {**balanced, 2: round(balanced[2] * 0.7, 3)}

    return Pipeline(
        [
            (
                "tfidf",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                ngram_range=cfg["word_ngram_range"],
                                min_df=cfg["word_min_df"],
                                max_features=cfg["word_max_features"],
                                sublinear_tf=cfg["word_sublinear_tf"],
                            ),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer=cfg["char_analyzer"],
                                ngram_range=cfg["char_ngram_range"],
                                min_df=cfg["char_min_df"],
                                max_features=cfg["char_max_features"],
                                sublinear_tf=cfg["char_sublinear_tf"],
                            ),
                        ),
                    ],
                    transformer_weights=cfg["union_transformer_weights"],
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=cfg["clf_C"],
                    class_weight=class_weight,
                    max_iter=cfg["clf_max_iter"],
                    solver=cfg["clf_solver"],
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = sorted(LABELS)
    f1 = f1_score(y_true, y_pred, average=None, labels=labels)
    precision = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    metrics = {
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    for idx, name in LABELS.items():
        metrics[f"f1_{name}"] = f1[idx]
        metrics[f"precision_{name}"] = precision[idx]
        metrics[f"recall_{name}"] = recall[idx]
    return {k: round(float(v), 4) for k, v in metrics.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("../Datasets/df.csv"))
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--config", choices=sorted(CONFIGS), default="exp4")
    parser.add_argument("--version", default="1", help="версия модели: tfidf-logreg@<version>")
    parser.add_argument("--max-rows", type=int, default=None, help="подвыборка для быстрого прогона")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    texts, labels, bad = load_fixed_csv(args.data)
    logger.info("прочитано %d записей из %s; пропущено строк: %s", len(texts), args.data, dict(bad) or 0)
    n_raw = len(texts)
    texts, labels, dedup_stats = deduplicate(texts, labels)
    logger.info("после очистки дублей: %d (%s)", len(texts), dedup_stats)

    if args.max_rows and args.max_rows < len(texts):
        texts, _, labels, _ = train_test_split(
            texts, labels, train_size=args.max_rows, random_state=RANDOM_STATE, stratify=labels
        )
        logger.info("подвыборка: %d записей", len(texts))

    y = np.asarray(labels, dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train = [clean_text(t) for t in X_train]
    X_test = [clean_text(t) for t in X_test]

    cfg = CONFIGS[args.config]
    model = build_pipeline(cfg, y_train)
    logger.info("обучение конфигурации %s на %d текстах...", args.config, len(X_train))
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0
    metrics = evaluate(y_test, model.predict(X_test))
    logger.info("обучено за %.0f с; метрики на отложенной выборке: %s", fit_time, metrics)

    args.out.mkdir(parents=True, exist_ok=True)
    model_path = args.out / "model.joblib"
    joblib.dump(model, model_path, compress=3)

    meta = {
        "name": "tfidf-logreg",
        "version": args.version,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "config_name": args.config,
        "config": {k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()},
        "metrics": metrics,
        "metrics_note": "Случайный стратифицированный сплит 80/20; оценка завышена (01-review R-36).",
        "fit_time_sec": round(fit_time, 1),
        "dataset": {
            "file": args.data.name,
            "sha256": sha256_of(args.data),
            "n_rows_raw": n_raw,
            "skipped_lines": dict(bad),
            **dedup_stats,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
        },
    }
    (args.out / "model_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("сохранено: %s, %s", model_path, args.out / "model_meta.json")


if __name__ == "__main__":
    main()
