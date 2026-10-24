"""Export saved RNN/CNN and the best saved LR, selecting blend weights on dev.

The notebook's original LR was not saved. Its refit was not numerically equivalent,
so this is a separately versioned ensemble of existing frozen model artifacts.
"""

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from app.artifacts import sha256_of
from app.ensemble import (
    CharCNN,
    Encoding,
    EnsemblePredictor,
    SpamClickbaitNet,
    blend,
    build_vocabularies,
    save_bundle,
)
from app.labels import CLASS_IDS, N_CLASSES
from app.logs import configure_logging
from app.text import clean_text
from training.data import import_dataset_scripts, load_fixed_csv
from training.metrics import evaluate

logger = logging.getLogger("export-ensemble")

# Компоненты, чьи вероятности на dev/test сохранил ноутбук (proba_<key>_<part>.npy)
CACHED = ("rnn", "char")
PARITY_SAMPLE = 64
SOURCE_NAMES = ["rnn_best.pt", "char_best.pt", "ensemble_weights.csv", "model_comparison.csv"]


@dataclass(frozen=True)
class Sources:
    """Сводка сплита и выбранного ноутбуком смешивания; пути к сохранённой LR и ноутбуку."""

    summary: dict
    weights_row: dict
    lr_path: Path
    notebook_path: Path


def restore_frame(data_path, filter_dataset):
    texts, labels, _ = load_fixed_csv(data_path)
    frame = pd.DataFrame({"text": texts, "is_spam1_or_clickbait2": labels})
    # Reuse the verified audit when available; split fingerprint below checks the
    # reconstructed texts/labels/order. Otherwise perform the same filters anew.
    audit = data_path.parent / "outputs/filtering/audit_2026-10-11"
    audit_summary = audit / "summary.json"
    if audit_summary.exists() and json.loads(audit_summary.read_text(encoding="utf-8")).get(
        "source_csv_sha256"
    ) == sha256_of(data_path):
        changes = pd.read_csv(
            audit / "changes.csv",
            usecols=["source_index", "action", "text_after"],
            encoding="utf-8-sig",
            keep_default_na=False,
        )
        corrected = changes.loc[changes.action.eq("corrected")]
        frame.loc[corrected.source_index, "text"] = corrected.text_after.to_numpy()
        frame = frame.drop(index=changes.loc[changes.action.eq("excluded"), "source_index"])
    else:
        frame, _, _ = filter_dataset(frame)
    frame = frame.drop_duplicates()
    counts = frame.groupby("text").is_spam1_or_clickbait2.nunique()
    return frame.loc[~frame.text.isin(counts[counts > 1].index)].reset_index(drop=True)


def check_close(actual, expected, name):
    if actual.shape != expected.shape:
        raise ValueError(f"{name}: probability shape mismatch")
    error = float(np.max(np.abs(actual - expected)))
    logger.info("parity %s: max absolute error %.9g", name, error)
    if not np.allclose(actual, expected, atol=5e-6, rtol=1e-5):
        raise ValueError(f"{name}: predictions differ from notebook (max error {error}); export aborted")
    return error


def select_weights(dev_parts, y_dev, step=0.1):
    """Перебор сетки весов на dev: первый максимум macro-F1 (строгое >), по rnn, затем по char."""
    n = round(1 / step)
    best_score, weights = -1, None
    for rnn_step in range(n + 1):
        for char_step in range(n + 1 - rnn_step):
            candidate = {"rnn": rnn_step / n, "char": char_step / n, "lr": (n - rnn_step - char_step) / n}
            pred = blend(dev_parts, candidate, normalize=False).argmax(1)
            score = float(f1_score(y_dev, pred, average="macro", labels=list(CLASS_IDS)))
            if score > best_score:
                best_score, weights = score, candidate
    return weights, best_score


def load_sources(args):
    """Сводки сплита, веса ноутбука и чекпойнты; все должны быть из одного fit/dev/test."""
    lr_path = args.lr_model or args.source.parent / "1_EDA-TFIDF_LogisticRegression/model.pkl"
    summary = json.loads((args.source / "split/summary.json").read_text(encoding="utf-8"))
    lr_summary = json.loads((lr_path.parent / "split/summary.json").read_text(encoding="utf-8"))
    if lr_summary["split_id"] != summary["split_id"]:
        raise ValueError("saved LR and neural models must have the same fit/dev/test")
    with (args.source / "ensemble_weights.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1 or rows[0]["split_id"] != summary["split_id"]:
        raise ValueError("ensemble weights and split are from different runs")
    checkpoints = {
        key: torch.load(args.source / f"{key}_best.pt", map_location="cpu", weights_only=False) for key in CACHED
    }
    for checkpoint in checkpoints.values():
        if checkpoint.get("split_id") != summary["split_id"] or checkpoint.get("evaluation_split") != "dev":
            raise ValueError("checkpoint must be selected on dev from the same split")
    sources = Sources(summary, rows[0], lr_path, args.data.parent / "2_RNN_CNN_LR.ipynb")
    return sources, checkpoints


def restore_models(fit_texts, checkpoints, lr_path):
    """Словари по fit, сети из чекпойнтов и сохранённая LR; веса смеси — временные, до select_weights."""
    logger.info("Restoring vocabularies from %d fit rows", len(fit_texts))
    encoding = Encoding()
    vocab, char_vocab = build_vocabularies(fit_texts, encoding)
    rnn_ckpt = checkpoints["rnn"]
    if len(vocab) != rnn_ckpt["vocab_size"] or len(char_vocab) != rnn_ckpt["char_vocab_size"]:
        raise ValueError("vocabulary dimensions differ from checkpoint")
    rnn = SpamClickbaitNet(len(vocab), len(char_vocab))
    char = CharCNN(len(char_vocab))
    rnn.load_state_dict(rnn_ckpt["model_state"])
    char.load_state_dict(checkpoints["char"]["model_state"])
    logger.info("Loading saved best LR: %s", lr_path)
    lr = joblib.load(lr_path)
    return EnsemblePredictor(rnn, char, lr, vocab, char_vocab, {"rnn": 0.4, "char": 0.4, "lr": 0.2}, encoding)


def verify_parity(predictor, texts, split, source):
    """Сети на случайной выборке dev/test дают те же вероятности, что сохранил ноутбук."""
    errors = {}
    for part in ("dev", "test"):
        rng = np.random.default_rng(42)
        positions = np.sort(rng.choice(len(split[part]), min(PARITY_SAMPLE, len(split[part])), replace=False))
        sample = texts.iloc[split[part][positions]].tolist()
        probabilities = predictor.component_probabilities(sample)
        for component in CACHED:
            expected = np.load(source / f"proba_{component}_{part}.npy", mmap_mode="r")[positions]
            errors[f"{component}_{part}"] = check_close(probabilities[component], expected, f"{component}/{part}")
    return errors


def split_parts(source, part, lr, texts, threads):
    """Вероятности компонентов на части сплита: сети — из кэша ноутбука, LR — заново."""
    parts = {key: np.load(source / f"proba_{key}_{part}.npy") for key in CACHED}
    with threadpool_limits(limits=threads):
        parts["lr"] = lr.predict_proba(texts)
    return parts


def evaluate_test(test_parts, weights, y_test):
    """Однократная оценка замороженной смеси на test."""
    return evaluate(y_test, blend(test_parts, weights).argmax(1), per_class=False, digits=None)


def build_meta(args, sources, lr, weights, dev_score, metrics, parity_errors):
    summary = sources.summary
    return {
        "name": "rnn-cnn-lr",
        "version": args.version,
        "framework": "PyTorch + scikit-learn",
        "trained_at": datetime.fromtimestamp((args.source / "rnn_best.pt").stat().st_mtime, UTC).isoformat(),
        "exported_at": datetime.now(UTC).isoformat(),
        "sklearn_version": sklearn.__version__,
        "torch_version": str(torch.__version__),
        "metrics": metrics,
        "dataset": {
            "file": args.data.name,
            "sha256": sha256_of(args.data),
            "split_id": summary["split_id"],
            "protocol": summary["protocol"],
            "partitions": summary["partitions"],
            "metrics_note": (
                "Measured test metrics for the new blend, using verified cached neural probabilities and saved LR. "
                "Near-duplicates/source overlap not fully eliminated; not ASR quality."
            ),
        },
        "config": {
            "weights": weights,
            "selection_split": "dev",
            "selection_metric": "f1_macro",
            "dev_f1_macro": dev_score,
            "original_notebook_dev_f1_macro": float(sources.weights_row["macro-F1 (валидация)"]),
            "lr_C": float(lr.named_steps["clf"].C),
            "normalizer": "notebook-clean-text-v1",
            "variant": "notebook-neural-models-plus-saved-best-lr",
            "lr_source_sha256": sha256_of(sources.lr_path),
            "parity": {"sample_rows_per_split": PARITY_SAMPLE, "max_absolute_errors": parity_errors},
            "source_notebook_sha256": sha256_of(sources.notebook_path),
            "source_artifact_sha256": {name: sha256_of(args.source / name) for name in SOURCE_NAMES},
        },
    }


def export(args):
    filter_dataset, make_data_splits = import_dataset_scripts(args.data)
    sources, checkpoints = load_sources(args)
    logger.info("Restoring filtered corpus and checking split identity")
    frame = restore_frame(args.data, filter_dataset)
    split = make_data_splits(frame, clean_text)
    if split["summary"]["split_id"] != sources.summary["split_id"]:
        raise ValueError("dataset/normalization/filtering changed since the recorded notebook run")
    texts = frame.text.map(clean_text)
    labels = frame.is_spam1_or_clickbait2
    predictor = restore_models(texts.iloc[split["fit"]].tolist(), checkpoints, sources.lr_path)
    del checkpoints  # веса уже в сетях; освобождаем память до инференса
    parity_errors = verify_parity(predictor, texts, split, args.source)

    logger.info("Selecting weights on dev with frozen models; no model refitting")
    dev_parts = split_parts(args.source, "dev", predictor.lr, texts.iloc[split["dev"]].tolist(), args.threads)
    y_dev = labels.iloc[split["dev"]].to_numpy()
    if any(value.shape != (len(y_dev), N_CLASSES) for value in dev_parts.values()):
        raise ValueError("dev probabilities do not match the split")
    weights, dev_score = select_weights(dev_parts, y_dev)
    predictor.weights = weights
    logger.info("Frozen weights %s, dev macro-F1 %.8f; evaluating test once", weights, dev_score)
    test_parts = split_parts(args.source, "test", predictor.lr, texts.iloc[split["test"]].tolist(), args.threads)
    metrics = evaluate_test(test_parts, weights, labels.iloc[split["test"]].to_numpy())

    meta = build_meta(args, sources, predictor.lr, weights, dev_score, metrics, parity_errors)
    path = save_bundle(args.out, predictor, meta)
    logger.info("EXPORTED %s; weights %s; test metrics %s (no models retrained)", path, weights, metrics)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("../Datasets/df.csv"))
    parser.add_argument("--source", type=Path, default=Path("../Datasets/outputs/validation/2_RNN_CNN_LR"))
    parser.add_argument("--out", type=Path, default=Path("models/ensemble"))
    parser.add_argument("--lr-model", type=Path, default=None, help="saved best LR with sibling split/summary.json")
    parser.add_argument("--version", default="ensemble-20261018")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    configure_logging()
    torch.set_num_threads(args.threads)
    export(args)


if __name__ == "__main__":
    main()
