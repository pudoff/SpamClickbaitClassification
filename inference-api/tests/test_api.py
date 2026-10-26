"""Тесты API через TestClient (шаг 5). Используют маленькую модель, обученную в фикстуре,
поэтому не требуют models/model.joblib и проходят за секунды."""

import json

import joblib
import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory, toy_lr):
    path = tmp_path_factory.mktemp("models")
    joblib.dump(toy_lr, path / "model.joblib")
    meta = {
        "name": "tfidf-logreg",
        "version": "test",
        "trained_at": "2026-10-09T10:00:00+00:00",
        "sklearn_version": "x",
        "metrics": {"f1_macro": 0.9, "accuracy": 0.91},
    }
    (path / "model_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def client(model_dir, make_settings):
    settings = make_settings(model_dir / "model.joblib", model_dir / "model_meta.json")
    with TestClient(create_app(settings)) as c:
        yield c


def test_healthcheck_ok(client):
    r = client.get("/healthcheck")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "model_version": "tfidf-logreg@test", "detail": None}


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["version"] == "test"
    assert body["trained_at"].startswith("2026-10-09")
    assert body["metrics"]["f1_macro"] == 0.9
    assert body["classes"] == {"0": "normal", "1": "spam", "2": "clickbait"}
    assert body["policy"] == {"tau": 0.55, "tau_spam": 0.65}


def test_predict_returns_valid_distribution(client):
    r = client.post("/predict", json={"text": "Вам одобрен кредит, перезвоните по номеру"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"classified", "uncertain"}
    assert abs(sum(body["scores"].values()) - 1) < 1e-6
    assert body["confidence"] == max(body["scores"].values())


def test_predict_threshold_zero_always_classifies(client):
    body = client.post("/predict", json={"text": "Вам одобрен кредит", "threshold": 0}).json()
    assert body["status"] == "classified"
    assert body["label"] == "spam" and body["label_id"] == 1


def test_low_confidence_is_uncertain_not_normal(client):
    body = client.post("/predict", json={"text": "Буду через десять минут", "threshold": 1}).json()
    assert body["status"] == "uncertain"
    assert body["label"] is None and body["reason"] == "low_confidence"


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "привет", "threshold": 1.5},  # Field(le=1)
        {"text": "привет", "threshold": -0.1},  # Field(ge=0)
        {"text": ""},  # min_length=1
        {"text": "123 !!! 😀"},  # нет букв
        {"text": "x" * 20_001},  # max_length
        {"text": "привет", "extra": 1},  # extra="forbid"
        {},  # нет обязательного поля
    ],
)
def test_validation_errors(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("payload", [{"text": "123 !!!"}, {"text": "привет", "threshold": 1.5}, {}])
def test_validation_precedes_unavailable_model(tmp_path, make_settings, payload):
    """Некорректное тело — 422 даже без модели: сначала валидация, потом проверка модели."""
    with TestClient(create_app(make_settings(tmp_path / "nope.joblib"))) as c:
        assert c.post("/predict", json=payload).status_code == 422


def test_service_without_model_reports_503(tmp_path, make_settings):
    with TestClient(create_app(make_settings(tmp_path / "nope.joblib", tmp_path / "nope.json"))) as c:
        health = c.get("/healthcheck")
        assert health.status_code == 503
        assert health.json()["model_loaded"] is False
        assert c.post("/predict", json={"text": "привет"}).status_code == 503
