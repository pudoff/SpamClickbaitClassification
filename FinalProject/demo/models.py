"""Конфигурации моделей, политика отказа и метрики.

Конфигурации — уменьшенные по числу признаков варианты BASE/EXP4 из
inference-api/training/train.py (для быстрого CPU-демо на выборке).
Политика decide повторяет семантику app/model.py: порог τ для обычного/кликбейта
и повышенный τ_spam для спама; ниже порога — uncertain, не normal.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import FeatureUnion, Pipeline

from demo.data import LABEL_NAMES

SEED = 42
TAU = 0.55
TAU_SPAM = 0.65

_WORD = dict(ngram_range=(1, 2), min_df=3, max_features=50_000, sublinear_tf=True)
_CHAR = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=50_000, sublinear_tf=True)

CONFIGS = {
    # E1: только словесные признаки
    "word_only": {"branches": ["word"], "union_weights": None, "C": 10.0},
    # E2: только символьные признаки
    "char_only": {"branches": ["char"], "union_weights": None, "C": 10.0},
    # E3: объединение word+char с весами (аналог exp4)
    "union_weighted": {"branches": ["word", "char"], "union_weights": {"word": 1.0, "char": 0.6}, "C": 10.0},
}


def build_pipeline(config_name: str) -> Pipeline:
    cfg = CONFIGS[config_name]
    branches = []
    if "word" in cfg["branches"]:
        branches.append(("word", TfidfVectorizer(**_WORD)))
    if "char" in cfg["branches"]:
        branches.append(("char", TfidfVectorizer(**_CHAR)))
    tfidf = branches[0][1] if len(branches) == 1 else FeatureUnion(branches, transformer_weights=cfg["union_weights"])
    return Pipeline(
        [
            ("tfidf", tfidf),
            ("clf", LogisticRegression(C=cfg["C"], class_weight="balanced", max_iter=1000, random_state=SEED)),
        ]
    )


def decide(proba: np.ndarray, tau: float = TAU, tau_spam: float = TAU_SPAM) -> list[dict]:
    """Политика отказа: список решений {status, label, confidence, reason}."""
    out = []
    for p in np.atleast_2d(proba):
        k = int(np.argmax(p))
        threshold = tau_spam if k == 1 else tau
        if p[k] >= threshold:
            out.append({"status": "classified", "label": LABEL_NAMES[k], "confidence": float(p[k]), "reason": None})
        else:
            out.append({"status": "uncertain", "label": None, "confidence": float(p[k]), "reason": "low_confidence"})
    return out


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = [0, 1, 2]
    pr, rc, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)), 4),
        "per_class": {
            LABEL_NAMES[c]: {
                "precision": round(float(pr[i]), 4),
                "recall": round(float(rc[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(sup[i]),
            }
            for i, c in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def policy_metrics(y_true: np.ndarray, proba: np.ndarray, tau: float = TAU, tau_spam: float = TAU_SPAM) -> dict:
    """Coverage и accuracy среди уверенных ответов (метрики политики отказа)."""
    decisions = decide(proba, tau, tau_spam)
    mask = np.array([d["status"] == "classified" for d in decisions])
    pred = np.argmax(proba, axis=1)
    n = len(y_true)
    return {
        "coverage": round(float(mask.mean()), 4) if n else 0.0,
        "n_classified": int(mask.sum()),
        "n_uncertain": int((~mask).sum()),
        "accuracy_classified": round(float(accuracy_score(y_true[mask], pred[mask])), 4) if mask.any() else None,
        "accuracy_all": round(float(accuracy_score(y_true, pred)), 4),
    }


def blend_proba(probas: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Взвешенная смесь вероятностей (ансамбль, как в rnn-cnn-lr)."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return sum(wi * p for wi, p in zip(w, probas))


def select_weights(dev_probas: list[np.ndarray], y_dev: np.ndarray, step: float = 0.1) -> tuple[list[float], float]:
    """Перебор весов смеси на dev по macro-F1 (шаг 0.1, сумма = 1)."""
    grid = np.arange(0, 1 + 1e-9, step)
    best_w, best_f1 = None, -1.0
    for w0 in grid:
        for w1 in grid:
            w2 = 1 - w0 - w1
            if w2 < -1e-9:
                continue
            w = [round(float(w0), 2), round(float(w1), 2), round(float(w2), 2)]
            p = blend_proba(dev_probas, w)
            f1 = f1_score(y_dev, p.argmax(axis=1), average="macro")
            if f1 > best_f1:
                best_w, best_f1 = w, float(f1)
    return best_w, best_f1