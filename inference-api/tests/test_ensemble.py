"""Экспортированный ансамбль через тот же загрузчик и эндпоинты, что в рабочем сервисе."""

import json
import shutil

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from app.ensemble import load_ensemble
from app.main import create_app
from app.text import clean_text
from tests.snapshot import snapshot


def test_bundle_roundtrip_and_weighted_probabilities(ensemble_bundle):
    path, original = ensemble_bundle
    restored, _ = load_ensemble(path)
    texts = [clean_text("Вам одобрен кредит срочно позвоните"), "", "неизвестноеслово", " NUM "]
    # Независимая взвешенная сумма до сохранения против публичного predict_proba после загрузки.
    parts = original.component_probabilities(texts)
    expected = 0.4 * parts["rnn"] + 0.4 * parts["char"] + 0.2 * parts["lr"]
    actual = restored.predict_proba(texts)
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual.sum(axis=1), 1)
    assert restored.predict_proba([]).shape == (0, 3)


SNAPSHOT_TEXTS = [
    "Вам одобрен кредит, перезвоните по номеру 8-800-555-35-35!",
    "Шок! Вы не поверите, что она сделала — смотрите до конца",
    "Буду через десять минут, возьми хлеба",
    "Скидка 90% только сегодня: https://example.com/sale",
    "",
]


def test_toy_ensemble_probabilities_snapshot(ensemble_bundle):
    """Вероятности игрушечного ансамбля до и после загрузки совпадают с сохранёнными (atol=1e-7)."""
    path, original = ensemble_bundle
    restored, _ = load_ensemble(path)
    texts = [clean_text(text) for text in SNAPSHOT_TEXTS]
    parts = original.component_probabilities(texts)
    actual = {"blend": original.predict_proba(texts).tolist(), **{key: value.tolist() for key, value in parts.items()}}
    expected = json.loads(snapshot("toy_ensemble_proba.json", actual))
    assert list(expected) == ["blend", "rnn", "char", "lr"]
    for key, value in expected.items():
        np.testing.assert_allclose(actual[key], value, rtol=0, atol=1e-7)
    np.testing.assert_allclose(restored.predict_proba(texts), expected["blend"], rtol=0, atol=1e-7)


def test_eval_does_not_update_batchnorm(ensemble_bundle):
    _, predictor = ensemble_bundle
    before = {key: value.clone() for key, value in predictor.rnn.state_dict().items()}
    predictor.predict_proba([clean_text("Буду через десять минут")])
    assert not predictor.rnn.training and not predictor.char.training
    for key, value in predictor.rnn.state_dict().items():
        assert torch.equal(value, before[key])


def test_ensemble_api_and_swagger(ensemble_bundle, make_settings):
    path, predictor = ensemble_bundle
    settings = make_settings(path)
    text = "Вам одобрен кредит срочно позвоните"
    expected = predictor.predict_proba([clean_text(text)])[0]
    with TestClient(create_app(settings)) as client:
        assert client.get("/healthcheck").status_code == 200
        info = client.get("/model-info").json()
        assert info["model_version"] == "rnn-cnn-lr@test"
        assert info["framework"] == "PyTorch + scikit-learn"
        assert info["config"]["weights"] == {"rnn": 0.4, "char": 0.4, "lr": 0.2}
        body = client.post("/predict", json={"text": text, "threshold": 0}).json()
        assert body["model_version"] == "rnn-cnn-lr@test"
        np.testing.assert_allclose([body["scores"][name] for name in ("normal", "spam", "clickbait")], expected)
        assert body["label_id"] == int(expected.argmax())
        assert client.get("/docs").status_code == 200
        assert {"/predict", "/healthcheck", "/model-info"}.issubset(client.get("/openapi.json").json()["paths"])


def corrupt_checksum(folder, manifest):
    (folder / "vocab.json").write_text("{}", encoding="utf-8")


def negative_weight(folder, manifest):
    manifest["weights"]["rnn"] = -0.4


def missing_file(folder, manifest):
    (folder / "char.pt").unlink()


def incompatible_sklearn(folder, manifest):
    manifest["meta"]["sklearn_version"] = "incompatible-version"


def wrong_class_order(folder, manifest):
    manifest["classes"] = [1, 0, 2]


@pytest.mark.parametrize(
    "damage",
    [corrupt_checksum, negative_weight, missing_file, incompatible_sklearn, wrong_class_order],
    ids=lambda damage: damage.__name__,
)
def test_invalid_ensemble_is_unavailable_without_lr_fallback(ensemble_bundle, tmp_path, make_settings, damage):
    path, _ = ensemble_bundle
    for artifact in path.parent.iterdir():
        shutil.copyfile(artifact, tmp_path / artifact.name)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    damage(tmp_path, manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with TestClient(create_app(make_settings(manifest_path))) as client:
        assert client.get("/healthcheck").status_code == 503
        assert client.post("/predict", json={"text": "Привет как дела"}).status_code == 503
