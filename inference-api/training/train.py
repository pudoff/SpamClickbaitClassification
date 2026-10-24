"""Обучение отдельной TF-IDF + LogisticRegression (tfidf-logreg) на Datasets/df.csv.

Сохраняет <out>/model.joblib и <out>/model_meta.json. Это эксперименты и запасной вариант сервиса;
рабочий ансамбль rnn-cnn-lr выпускается скриптом training/export_ensemble.py.

    python -m training.train --data=../Datasets/df.csv --out=models --config=exp4
"""

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from app.artifacts import sha256_of
from app.logs import configure_logging
from app.text import clean_text
from training.data import deduplicate, load_fixed_csv
from training.metrics import evaluate

logger = logging.getLogger("train")

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
    # Лучший прогон в Datasets/outputs/mlflow.db: run_7_best_confusion_matrix -> best_run=run_5_exp4_union_weights
    "exp4": EXP4_CONFIG,
    "exp5": EXP5_CONFIG,
}


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("../Datasets/df.csv"))
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--config", choices=sorted(CONFIGS), default="exp4")
    parser.add_argument("--version", default="1", help="версия модели: tfidf-logreg@<version>")
    parser.add_argument("--max-rows", type=int, default=None, help="подвыборка для быстрого прогона")
    args = parser.parse_args()

    configure_logging()

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
