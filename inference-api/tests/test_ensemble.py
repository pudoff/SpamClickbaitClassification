"""Exercise exported ensembles through the same loader and endpoints as production."""

import json

import numpy as np
import pytest
import torch
from app.config import Settings
from app.ensemble import (
    CharCNN,
    Encoding,
    EnsemblePredictor,
    SpamClickbaitNet,
    build_vocabularies,
    load_ensemble,
    save_bundle,
)
from app.main import create_app
from app.text import clean_text
from fastapi.testclient import TestClient
from training.train import BASE_CONFIG, build_pipeline

from tests.test_api import TOY_DATA


@pytest.fixture(scope="module")
def ensemble_bundle(tmp_path_factory):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    texts = [clean_text(text) for text, _ in TOY_DATA]
    labels = np.array([label for _, label in TOY_DATA])
    encoding = Encoding(max_len=16, max_chars=24, vocab_limit=100, min_freq=1)
    vocab, char_vocab = build_vocabularies(texts, encoding)
    lr = build_pipeline({**BASE_CONFIG, "word_min_df": 1, "char_min_df": 1}, labels).fit(texts, labels)
    predictor = EnsemblePredictor(
        SpamClickbaitNet(len(vocab), len(char_vocab)),
        CharCNN(len(char_vocab)),
        lr,
        vocab,
        char_vocab,
        {"rnn": 0.4, "char": 0.4, "lr": 0.2},
        encoding,
    )
    folder = tmp_path_factory.mktemp("ensemble")
    path = save_bundle(
        folder,
        predictor,
        {
            "name": "rnn-cnn-lr",
            "version": "test",
            "framework": "PyTorch + scikit-learn",
            "config": {"weights": predictor.weights, "selection_split": "dev"},
        },
    )
    return path, predictor


def test_bundle_roundtrip_and_weighted_probabilities(ensemble_bundle):
    path, original = ensemble_bundle
    restored, _ = load_ensemble(path)
    texts = [clean_text("Вам одобрен кредит срочно позвоните"), "", "неизвестноеслово", " NUM "]
    # Independent weighted sum before serialization vs public predictor after loading.
    parts = original.component_probabilities(texts)
    expected = 0.4 * parts["rnn"] + 0.4 * parts["char"] + 0.2 * parts["lr"]
    actual = restored.predict_proba(texts)
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual.sum(axis=1), 1)
    assert restored.predict_proba([]).shape == (0, 3)


def test_eval_does_not_update_batchnorm(ensemble_bundle):
    _, predictor = ensemble_bundle
    before = {key: value.clone() for key, value in predictor.rnn.state_dict().items()}
    predictor.predict_proba([clean_text("Буду через десять минут")])
    assert not predictor.rnn.training and not predictor.char.training
    for key, value in predictor.rnn.state_dict().items():
        assert torch.equal(value, before[key])


def test_ensemble_api_and_swagger(ensemble_bundle):
    path, predictor = ensemble_bundle
    settings = Settings(model_path=path, meta_path=path.with_name("unused.json"), tau=0.55, tau_spam=0.65)
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


@pytest.mark.parametrize("damage", ["checksum", "weights", "missing", "sklearn", "classes"])
def test_invalid_ensemble_is_unavailable_without_lr_fallback(ensemble_bundle, tmp_path, damage):
    import shutil

    path, _ = ensemble_bundle
    for artifact in path.parent.iterdir():
        shutil.copyfile(artifact, tmp_path / artifact.name)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if damage == "checksum":
        (tmp_path / "vocab.json").write_text("{}", encoding="utf-8")
    elif damage == "weights":
        manifest["weights"]["rnn"] = -0.4
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif damage == "sklearn":
        manifest["meta"]["sklearn_version"] = "incompatible-version"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif damage == "classes":
        manifest["classes"] = [1, 0, 2]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        (tmp_path / "char.pt").unlink()
    settings = Settings(model_path=manifest_path, meta_path=tmp_path / "unused.json", tau=0.55, tau_spam=0.65)
    with TestClient(create_app(settings)) as client:
        assert client.get("/healthcheck").status_code == 503
        assert client.post("/predict", json={"text": "Привет как дела"}).status_code == 503
