"""Шаги экспорта ансамбля на синтетических массивах (без артефактов ноутбуков)."""

import argparse
import json

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import accuracy_score, f1_score

from app.artifacts import sha256_of
from app.ensemble import CharCNN, Encoding, SpamClickbaitNet, blend, build_vocabularies
from tests.snapshot import MANIFEST_SNAPSHOT
from training.export_ensemble import (
    build_meta,
    evaluate_test,
    load_sources,
    restore_models,
    select_weights,
    verify_parity,
)
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


META_KEYS = [
    "name",
    "version",
    "framework",
    "trained_at",
    "exported_at",
    "sklearn_version",
    "torch_version",
    "metrics",
    "dataset",
    "config",
]
DATASET_KEYS = ["file", "sha256", "split_id", "protocol", "partitions", "metrics_note"]
CONFIG_KEYS = [
    "weights",
    "selection_split",
    "selection_metric",
    "dev_f1_macro",
    "original_notebook_dev_f1_macro",
    "lr_C",
    "normalizer",
    "variant",
    "lr_source_sha256",
    "parity",
    "source_notebook_sha256",
    "source_artifact_sha256",
]


@pytest.fixture
def layout(tmp_path, toy_texts, toy_lr):
    """Каталоги как у ноутбуков: Datasets/df.csv, outputs/validation/{2_RNN_CNN_LR, 1_EDA-…}."""
    data = tmp_path / "Datasets/df.csv"
    source = tmp_path / "Datasets/outputs/validation/2_RNN_CNN_LR"
    lr_dir = source.parent / "1_EDA-TFIDF_LogisticRegression"
    for folder in (source / "split", lr_dir / "split"):
        folder.mkdir(parents=True)
    data.write_text("text,is_spam1_or_clickbait2\nпривет,0\n", encoding="utf-8")
    (data.parent / "2_RNN_CNN_LR.ipynb").write_text("{}", encoding="utf-8")
    summary = {"split_id": "abc", "protocol": "grouped", "partitions": {"fit": {"rows": 1}}}
    for folder in (source, lr_dir):
        (folder / "split/summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (source / "ensemble_weights.csv").write_text("split_id,macro-F1 (валидация)\nabc,0.81\n", encoding="utf-8-sig")
    (source / "model_comparison.csv").write_text("model\n", encoding="utf-8")
    joblib.dump(toy_lr, lr_dir / "model.pkl")
    vocab, char_vocab = build_vocabularies(toy_texts, Encoding())
    torch.manual_seed(0)
    nets = {"rnn": SpamClickbaitNet(len(vocab), len(char_vocab)), "char": CharCNN(len(char_vocab))}
    for key, net in nets.items():
        checkpoint = {"split_id": "abc", "evaluation_split": "dev", "model_state": net.state_dict()}
        checkpoint |= {"vocab_size": len(vocab), "char_vocab_size": len(char_vocab)}
        torch.save(checkpoint, source / f"{key}_best.pt")
    args = argparse.Namespace(
        data=data, source=source, lr_model=None, version="ensemble-test", threads=1, normalizer="notebook-clean-text-v1"
    )
    return args, nets


def test_load_sources(layout):
    args, _ = layout
    sources, checkpoints = load_sources(args)
    assert sources.summary["split_id"] == "abc" and sources.weights_row["macro-F1 (валидация)"] == "0.81"
    assert sources.lr_path.name == "model.pkl" and sources.notebook_path.name == "2_RNN_CNN_LR.ipynb"
    assert list(checkpoints) == ["rnn", "char"]


def other_lr_split(args):
    path = args.source.parent / "1_EDA-TFIDF_LogisticRegression/split/summary.json"
    path.write_text('{"split_id": "other"}', encoding="utf-8")


def other_weights_run(args):
    (args.source / "ensemble_weights.csv").write_text("split_id,macro-F1 (валидация)\nother,0.8\n", encoding="utf-8")


def checkpoint_not_on_dev(args):
    checkpoint = torch.load(args.source / "char_best.pt", weights_only=False)
    torch.save({**checkpoint, "evaluation_split": "test"}, args.source / "char_best.pt")


@pytest.mark.parametrize(
    "damage", [other_lr_split, other_weights_run, checkpoint_not_on_dev], ids=lambda damage: damage.__name__
)
def test_load_sources_rejects_mixed_runs(layout, damage):
    args, _ = layout
    damage(args)
    with pytest.raises(ValueError):
        load_sources(args)


def test_restore_models_and_parity(layout, toy_texts):
    args, nets = layout
    sources, checkpoints = load_sources(args)
    predictor = restore_models(toy_texts, checkpoints, sources.lr_path)
    for key, net in nets.items():
        restored = getattr(predictor, key).state_dict()
        assert all(torch.equal(value, restored[name]) for name, value in net.state_dict().items())
    texts = pd.Series(toy_texts)
    split = {"dev": np.arange(0, 20), "test": np.arange(20, len(toy_texts))}
    for part, rows in split.items():
        parts = predictor.component_probabilities(texts.iloc[rows].tolist())
        for key in ("rnn", "char"):
            np.save(args.source / f"proba_{key}_{part}.npy", parts[key])
    errors = verify_parity(predictor, texts, split, args.source)
    assert list(errors) == ["rnn_dev", "char_dev", "rnn_test", "char_test"]
    assert max(errors.values()) < 1e-6
    np.save(args.source / "proba_char_test.npy", np.full((len(split["test"]), 3), 1 / 3))
    with pytest.raises(ValueError):
        verify_parity(predictor, texts, split, args.source)


def test_evaluate_test(parts):
    y_test = np.random.default_rng(5).integers(0, 3, size=200)
    weights = {"rnn": 0.5, "char": 0.2, "lr": 0.3}
    metrics = evaluate_test(parts, weights, y_test)
    assert list(metrics) == ["accuracy", "f1_macro", "f1_weighted"]
    assert metrics["accuracy"] == float(accuracy_score(y_test, blend(parts, weights).argmax(1)))


def test_build_meta_structure(layout, toy_lr):
    args, _ = layout
    sources, _ = load_sources(args)
    weights = {"rnn": 0.5, "char": 0.2, "lr": 0.3}
    metrics = {"accuracy": 0.9, "f1_macro": 0.8, "f1_weighted": 0.85}
    meta = build_meta(args, sources, toy_lr, weights, 0.83, metrics, {"rnn_dev": 0.0})
    assert list(meta) == META_KEYS
    assert list(meta["dataset"]) == DATASET_KEYS
    assert list(meta["config"]) == CONFIG_KEYS
    assert (meta["name"], meta["version"], meta["metrics"]) == ("rnn-cnn-lr", "ensemble-test", metrics)
    assert meta["config"]["original_notebook_dev_f1_macro"] == 0.81
    assert meta["config"]["lr_C"] == 10.0 and meta["config"]["dev_f1_macro"] == 0.83
    assert meta["config"]["parity"] == {"sample_rows_per_split": 64, "max_absolute_errors": {"rnn_dev": 0.0}}
    assert list(meta["config"]["source_artifact_sha256"]) == [
        "rnn_best.pt",
        "char_best.pt",
        "ensemble_weights.csv",
        "model_comparison.csv",
    ]
    assert meta["dataset"]["sha256"] == sha256_of(args.data)


@pytest.mark.skipif(not MANIFEST_SNAPSHOT.exists(), reason="Reports/ не смонтирован (запуск в образе)")
def test_meta_keys_match_reference_manifest():
    meta = json.loads(MANIFEST_SNAPSHOT.read_text(encoding="utf-8"))["meta"]
    assert (list(meta), list(meta["dataset"]), list(meta["config"])) == (META_KEYS, DATASET_KEYS, CONFIG_KEYS)
