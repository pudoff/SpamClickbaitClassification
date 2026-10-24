"""Характеризационные тесты: фиксируют текущее поведение контракта, политики решения и форматов."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.model import decide
from tests.snapshot import dump, snapshot
from training.train import evaluate

MANIFEST_SNAPSHOT = Path(__file__).resolve().parents[2] / "Reports/2026-10-18/model_card/model_manifest_snapshot.json"
MANIFEST_KEYS = ["format", "classes", "weights", "encoding", "files", "meta"]
# Буквально, а не из app.ensemble: тест ловит и случайную правку констант.
BUNDLE_FILES = {"rnn": "rnn.pt", "char": "char.pt", "lr": "lr.joblib", "vocab": "vocab.json"}
ENCODING_KEYS = ["max_len", "max_chars", "vocab_limit", "min_freq"]


def test_openapi_matches_snapshot(tmp_path):
    settings = Settings(model_path=tmp_path / "nope.json", meta_path=tmp_path / "nope.json", tau=0.55, tau_spam=0.65)
    schema = TestClient(create_app(settings)).get("/openapi.json").json()
    assert dump(schema) == snapshot("openapi.json", schema)


def _fields(decision) -> dict:
    data = dataclasses.asdict(decision) if dataclasses.is_dataclass(decision) else dict(decision)
    return {key: data[key] for key in ("status", "label", "label_id", "confidence", "threshold", "reason")}


@pytest.mark.parametrize(
    ("scores", "threshold", "expected"),
    [
        # threshold=None: для spam берётся TAU_SPAM, для остальных — TAU
        ({"normal": 0.2, "spam": 0.6, "clickbait": 0.2}, None, ("uncertain", None, 0.6, 0.65)),
        ({"normal": 0.2, "spam": 0.7, "clickbait": 0.1}, None, ("classified", "spam", 0.7, 0.65)),
        ({"normal": 0.2, "spam": 0.2, "clickbait": 0.6}, None, ("classified", "clickbait", 0.6, 0.55)),
        ({"normal": 0.5, "spam": 0.3, "clickbait": 0.2}, None, ("uncertain", None, 0.5, 0.55)),
        # граница: confidence == threshold -> classified
        ({"normal": 0.55, "spam": 0.25, "clickbait": 0.2}, None, ("classified", "normal", 0.55, 0.55)),
        ({"normal": 0.15, "spam": 0.65, "clickbait": 0.2}, None, ("classified", "spam", 0.65, 0.65)),
        # явный threshold заменяет политику для любого класса
        ({"normal": 0.1, "spam": 0.8, "clickbait": 0.1}, 0.9, ("uncertain", None, 0.8, 0.9)),
        ({"normal": 0.1, "spam": 0.6, "clickbait": 0.3}, 0.6, ("classified", "spam", 0.6, 0.6)),
        ({"normal": 0.4, "spam": 0.3, "clickbait": 0.3}, 0.0, ("classified", "normal", 0.4, 0.0)),
        # низкая уверенность никогда не превращается в normal
        ({"normal": 0.99, "spam": 0.005, "clickbait": 0.005}, 1.0, ("uncertain", None, 0.99, 1.0)),
        # ничья: побеждает первый класс в порядке scores
        ({"normal": 0.4, "spam": 0.4, "clickbait": 0.2}, 0.0, ("classified", "normal", 0.4, 0.0)),
    ],
)
def test_decide_table(scores, threshold, expected):
    status, label, confidence, used_threshold = expected
    reason = None if status == "classified" else "low_confidence"
    assert _fields(decide(scores, 0.55, 0.65, threshold)) == {
        "status": status,
        "label": label,
        "label_id": {None: None, "normal": 0, "spam": 1, "clickbait": 2}[label],
        "confidence": confidence,
        "threshold": used_threshold,
        "reason": reason,
    }


def test_train_evaluate_exact():
    y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 0, 1, 1, 2, 2, 0, 2, 0, 1, 1])
    assert list(evaluate(y_true, y_pred).items()) == [
        ("f1_macro", 0.6627),
        ("f1_weighted", 0.6627),
        ("accuracy", 0.6667),
        ("f1_normal", 0.75),
        ("precision_normal", 0.75),
        ("recall_normal", 0.75),
        ("f1_spam", 0.6667),
        ("precision_spam", 0.6),
        ("recall_spam", 0.75),
        ("f1_clickbait", 0.5714),
        ("precision_clickbait", 0.6667),
        ("recall_clickbait", 0.5),
    ]


def assert_manifest_structure(manifest: dict) -> None:
    assert list(manifest) == MANIFEST_KEYS
    assert manifest["format"] == "spam-clickbait-ensemble-v1"
    assert manifest["classes"] == [0, 1, 2]
    assert list(manifest["weights"]) == ["rnn", "char", "lr"]
    assert list(manifest["encoding"]) == ENCODING_KEYS
    assert list(manifest["files"]) == list(BUNDLE_FILES)
    for key, name in BUNDLE_FILES.items():
        assert list(manifest["files"][key]) == ["path", "sha256"]
        assert manifest["files"][key]["path"] == name


def test_saved_manifest_structure(ensemble_bundle):
    path, _ = ensemble_bundle
    assert_manifest_structure(json.loads(path.read_text(encoding="utf-8")))


@pytest.mark.skipif(not MANIFEST_SNAPSHOT.exists(), reason="Reports/ не смонтирован (запуск в образе)")
def test_reference_manifest_structure():
    assert_manifest_structure(json.loads(MANIFEST_SNAPSHOT.read_text(encoding="utf-8")))
