"""Тесты API через TestClient (шаг 5). Используют маленькую модель, обученную в фикстуре,
поэтому не требуют models/model.joblib и проходят за секунды."""

import json

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.text import clean_text
from training.train import BASE_CONFIG, build_pipeline

TOY_DATA = [
    ("Вам одобрен кредит, перезвоните по номеру и получите деньги сегодня", 1),
    ("Скидка 90 процентов только сегодня купите сейчас переходите по ссылке", 1),
    ("Выиграйте приз отправьте данные карты для получения выигрыша", 1),
    ("Ты не поверишь что случилось со звездой шок фото", 2),
    ("Шок вы не поверите что она сделала смотрите до конца", 2),
    ("Учёные в шоке никто не ожидал такого финала", 2),
    ("Буду через десять минут возьми хлеба", 0),
    ("Привет как дела давай встретимся завтра вечером", 0),
    ("Отчёт отправил посмотри когда будет время", 0),
] * 4


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("models")
    texts = [clean_text(t) for t, _ in TOY_DATA]
    y = np.array([lb for _, lb in TOY_DATA])
    cfg = {**BASE_CONFIG, "word_min_df": 1, "char_min_df": 1}
    model = build_pipeline(cfg, y).fit(texts, y)
    joblib.dump(model, path / "model.joblib")
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
def client(model_dir):
    settings = Settings(
        model_path=model_dir / "model.joblib", meta_path=model_dir / "model_meta.json", tau=0.55, tau_spam=0.65
    )
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


def test_service_without_model_reports_503(tmp_path):
    settings = Settings(model_path=tmp_path / "nope.joblib", meta_path=tmp_path / "nope.json", tau=0.55, tau_spam=0.65)
    with TestClient(create_app(settings)) as c:
        health = c.get("/healthcheck")
        assert health.status_code == 503
        assert health.json()["model_loaded"] is False
        assert c.post("/predict", json={"text": "привет"}).status_code == 503
