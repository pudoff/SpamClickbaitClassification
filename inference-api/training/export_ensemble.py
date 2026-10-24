"""Export saved RNN/CNN and the best saved LR, selecting blend weights on dev.

The notebook's original LR was not saved. Its refit was not numerically equivalent,
so this is a separately versioned ensemble of existing frozen model artifacts.
"""

import argparse
import csv
import json
import logging
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
        frame, changes, _ = filter_dataset(frame)
    del changes
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


def export(args):
    filter_dataset, make_data_splits = import_dataset_scripts(args.data)
    notebook_path = args.data.parent / "2_RNN_CNN_LR.ipynb"
    lr_path = args.lr_model or args.source.parent / "1_EDA-TFIDF_LogisticRegression/model.pkl"
    summary = json.loads((args.source / "split/summary.json").read_text(encoding="utf-8"))
    lr_summary = json.loads((lr_path.parent / "split/summary.json").read_text(encoding="utf-8"))
    if lr_summary["split_id"] != summary["split_id"]:
        raise ValueError("saved LR and neural models must have the same fit/dev/test")
    with (args.source / "ensemble_weights.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1 or rows[0]["split_id"] != summary["split_id"]:
        raise ValueError("ensemble weights and split are from different runs")
    weights_row = rows[0]
    rnn_ckpt = torch.load(args.source / "rnn_best.pt", map_location="cpu", weights_only=False)
    char_ckpt = torch.load(args.source / "char_best.pt", map_location="cpu", weights_only=False)
    for checkpoint in (rnn_ckpt, char_ckpt):
        if checkpoint.get("split_id") != summary["split_id"] or checkpoint.get("evaluation_split") != "dev":
            raise ValueError("checkpoint must be selected on dev from the same split")
    logger.info("Restoring filtered corpus and checking split identity")
    frame = restore_frame(args.data, filter_dataset)
    split = make_data_splits(frame, clean_text)
    if split["summary"]["split_id"] != summary["split_id"]:
        raise ValueError("dataset/normalization/filtering changed since the recorded notebook run")
    texts = frame.text.map(clean_text)
    fit_texts = texts.iloc[split["fit"]].tolist()
    logger.info("Restoring vocabularies from %d fit rows", len(fit_texts))
    encoding = Encoding()
    vocab, char_vocab = build_vocabularies(fit_texts, encoding)
    if len(vocab) != rnn_ckpt["vocab_size"] or len(char_vocab) != rnn_ckpt["char_vocab_size"]:
        raise ValueError("vocabulary dimensions differ from checkpoint")
    rnn = SpamClickbaitNet(len(vocab), len(char_vocab))
    char = CharCNN(len(char_vocab))
    rnn.load_state_dict(rnn_ckpt["model_state"])
    char.load_state_dict(char_ckpt["model_state"])
    del rnn_ckpt, char_ckpt
    logger.info("Loading saved best LR: %s", lr_path)
    lr = joblib.load(lr_path)
    predictor = EnsemblePredictor(rnn, char, lr, vocab, char_vocab, {"rnn": 0.4, "char": 0.4, "lr": 0.2}, encoding)
    errors = {}
    sample_count = 64
    for part in ("dev", "test"):
        rng = np.random.default_rng(42)
        positions = np.sort(rng.choice(len(split[part]), min(sample_count, len(split[part])), replace=False))
        sample = texts.iloc[split[part][positions]].tolist()
        probabilities = predictor.component_probabilities(sample)
        for component in ("rnn", "char"):
            expected = np.load(args.source / f"proba_{component}_{part}.npy", mmap_mode="r")[positions]
            errors[f"{component}_{part}"] = check_close(probabilities[component], expected, f"{component}/{part}")
    logger.info("Selecting weights on dev with frozen models; no model refitting")
    dev_parts = {key: np.load(args.source / f"proba_{key}_dev.npy") for key in ("rnn", "char")}
    dev_texts = texts.iloc[split["dev"]].tolist()
    with threadpool_limits(limits=args.threads):
        dev_parts["lr"] = lr.predict_proba(dev_texts)
    y_dev = frame.iloc[split["dev"]].is_spam1_or_clickbait2.to_numpy()
    if any(value.shape != (len(y_dev), N_CLASSES) for value in dev_parts.values()):
        raise ValueError("dev probabilities do not match the split")
    weights, best_score = select_weights(dev_parts, y_dev)
    predictor.weights = weights
    logger.info("Frozen weights %s, dev macro-F1 %.8f; evaluating test once", weights, best_score)
    test_parts = {key: np.load(args.source / f"proba_{key}_test.npy") for key in ("rnn", "char")}
    with threadpool_limits(limits=args.threads):
        test_parts["lr"] = lr.predict_proba(texts.iloc[split["test"]].tolist())
    probabilities = blend(test_parts, weights)
    y_test = frame.iloc[split["test"]].is_spam1_or_clickbait2.to_numpy()
    metrics = evaluate(y_test, probabilities.argmax(1), per_class=False, digits=None)
    source_names = ["rnn_best.pt", "char_best.pt", "ensemble_weights.csv", "model_comparison.csv"]
    meta = {
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
            "dev_f1_macro": best_score,
            "original_notebook_dev_f1_macro": float(weights_row["macro-F1 (валидация)"]),
            "lr_C": float(lr.named_steps["clf"].C),
            "normalizer": "notebook-clean-text-v1",
            "variant": "notebook-neural-models-plus-saved-best-lr",
            "lr_source_sha256": sha256_of(lr_path),
            "parity": {"sample_rows_per_split": sample_count, "max_absolute_errors": errors},
            "source_notebook_sha256": sha256_of(notebook_path),
            "source_artifact_sha256": {name: sha256_of(args.source / name) for name in source_names},
        },
    }
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
