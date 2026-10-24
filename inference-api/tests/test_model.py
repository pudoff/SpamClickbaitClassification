"""Загрузка модели и метаданные LoadedModel."""

import pickle

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

from app.main import create_app
from app.model import load_model
from app.text import clean_text, clean_text_v2


def dummy(classes):
    return DummyClassifier(strategy="prior").fit(np.zeros((len(classes), 1)), classes)


def test_lr_without_meta_gets_defaults(tmp_path):
    path = tmp_path / "model.pkl"  # артефакт ноутбука: pickle.dump
    with path.open("wb") as f:
        pickle.dump(dummy([0, 1, 2]), f)
    model = load_model(path, tmp_path / "missing.json")
    assert (model.name, model.version, model.model_version) == ("tfidf-logreg", "0", "tfidf-logreg@0")
    assert model.trained_at is not None and model.trained_at.tzinfo is not None
    assert model.classes == [0, 1, 2] and len(model.sha256) == 64


def test_trained_at_is_optional(tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump(dummy([0, 1, 2]), path)
    (tmp_path / "meta.json").write_text('{"name": "x", "version": 3}', encoding="utf-8")
    model = load_model(path, tmp_path / "meta.json")
    assert (model.model_version, model.trained_at) == ("x@3", None)


def test_wrong_classes_rejected(tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump(dummy([0, 1]), path)
    with pytest.raises(ValueError, match="классы"):
        load_model(path, tmp_path / "missing.json")


def test_unloaded_model_bodies(tmp_path, make_settings):
    joblib.dump(dummy([0, 1]), tmp_path / "model.joblib")
    with TestClient(create_app(make_settings(tmp_path / "model.joblib"))) as client:
        health = client.get("/healthcheck")
        assert health.status_code == 503
        body = health.json()
        assert (body["status"], body["model_loaded"], body["model_version"]) == ("unavailable", False, None)
        assert body["detail"].startswith("ValueError: модель знает классы")
        for response in (client.post("/predict", json={"text": "привет"}), client.get("/model-info")):
            assert response.status_code == 503
            assert response.json()["detail"].startswith("модель не загружена: ValueError")


def test_normalizer_comes_from_meta(tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump(dummy([0, 1, 2]), path)
    legacy = load_model(path, tmp_path / "missing.json")
    assert legacy.normalizer == "notebook-clean-text-v1" and legacy.normalize is clean_text
    (tmp_path / "meta.json").write_text('{"config": {"normalizer": "clean-text-v2"}}', encoding="utf-8")
    assert load_model(path, tmp_path / "meta.json").normalize is clean_text_v2


def test_unknown_normalizer_is_unavailable(tmp_path, make_settings):
    joblib.dump(dummy([0, 1, 2]), tmp_path / "model.joblib")
    (tmp_path / "meta.json").write_text('{"config": {"normalizer": "v3"}}', encoding="utf-8")
    with TestClient(create_app(make_settings(tmp_path / "model.joblib", tmp_path / "meta.json"))) as client:
        health = client.get("/healthcheck")
        assert health.status_code == 503 and "нормализация" in health.json()["detail"]
