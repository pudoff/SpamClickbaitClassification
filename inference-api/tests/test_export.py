"""Шаги экспорта ансамбля на синтетических массивах (без артефактов ноутбуков)."""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

from app.ensemble import blend
from training.export_ensemble import select_weights
from training.metrics import evaluate


@pytest.fixture
def parts():
    rng = np.random.default_rng(7)
    return {key: rng.dirichlet(np.ones(3), size=200) for key in ("rnn", "char", "lr")}


def legacy_select_weights(dev_parts, y_dev):
    """Перебор весов в прежнем виде (до рефакторинга) — эталон для сравнения."""
    best_score, weights = -1, None
    for rnn_step in range(11):
        for char_step in range(11 - rnn_step):
            candidate = {"rnn": rnn_step / 10, "char": char_step / 10, "lr": (10 - rnn_step - char_step) / 10}
            mixed = sum(candidate[key] * dev_parts[key] for key in candidate)
            score = float(f1_score(y_dev, mixed.argmax(1), average="macro", labels=[0, 1, 2]))
            if score > best_score:
                best_score, weights = score, candidate
    return weights, best_score


def test_blend_matches_manual_sum(parts):
    weights = {"rnn": 0.5, "char": 0.2, "lr": 0.3}
    raw = 0.5 * parts["rnn"] + 0.2 * parts["char"] + 0.3 * parts["lr"]
    np.testing.assert_array_equal(blend(parts, weights, normalize=False), raw)
    np.testing.assert_array_equal(blend(parts, weights), raw / raw.sum(axis=1, keepdims=True))


def test_select_weights_matches_legacy_loop(parts):
    y_dev = np.random.default_rng(3).integers(0, 3, size=200)
    assert select_weights(parts, y_dev) == legacy_select_weights(parts, y_dev)


def test_select_weights_takes_first_maximum():
    # Все компоненты одинаковы: любая смесь даёт одинаковый F1, побеждает первая (rnn=0, char=0, lr=1).
    same = np.eye(3)[[0, 1, 2, 0]]
    weights, score = select_weights({"rnn": same, "char": same, "lr": same}, np.array([0, 1, 2, 0]))
    assert weights == {"rnn": 0.0, "char": 0.0, "lr": 1.0}
    assert score == 1.0


def test_export_metrics_match_legacy_dict():
    rng = np.random.default_rng(11)
    y_true, y_pred = rng.integers(0, 3, size=300), rng.integers(0, 3, size=300)
    legacy = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2])),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", labels=[0, 1, 2])),
    }
    assert list(evaluate(y_true, y_pred, per_class=False, digits=None).items()) == list(legacy.items())
