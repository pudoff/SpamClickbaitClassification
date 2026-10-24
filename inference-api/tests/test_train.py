"""Тренер LR через CLI на маленьком CSV: артефакты, метаданные и загрузка сервисом."""

import json
import pickle
import sys

import pandas as pd
from fastapi.testclient import TestClient

from app.main import create_app
from app.text import NORMALIZER_V2, clean_text_v2
from tests.conftest import TOY_DATA
from training import train

META_KEYS = [
    "name",
    "version",
    "trained_at",
    "sklearn_version",
    "config_name",
    "config",
    "metrics",
    "metrics_note",
    "fit_time_sec",
    "dataset",
]
DATASET_KEYS = [
    "file",
    "sha256",
    "n_rows_raw",
    "skipped_lines",
    "duplicates_removed",
    "conflicting_texts_removed",
    "n_train",
    "n_test",
    "random_state",
    "test_size",
]


def test_train_cli_end_to_end(tmp_path, monkeypatch, make_settings):
    rows = [(f"{text} ВАРИАНТ HTTP://X.RU/{i}", label) for text, label in TOY_DATA[:9] for i in range(6)]
    data = tmp_path / "df.csv"
    data.write_text("text,is_spam1_or_clickbait2\n" + "".join(f'"{t}",{lb}\n' for t, lb in rows), encoding="utf-8")
    out = tmp_path / "models"
    argv = ["train", f"--data={data}", f"--out={out}", "--config=baseline", "--version=t", "--test-sample"]
    monkeypatch.setattr(sys, "argv", argv)
    train.main()

    meta = json.loads((out / "model_meta.json").read_text(encoding="utf-8"))
    assert list(meta) == META_KEYS and list(meta["dataset"]) == DATASET_KEYS
    assert meta["config"]["normalizer"] == NORMALIZER_V2 and list(meta["config"])[-1] == "normalizer"
    assert all(value == round(value, 4) for value in meta["metrics"].values())

    with (out / "test_sample.pkl").open("rb") as f:
        sample = pickle.load(f)
    assert isinstance(sample, pd.DataFrame) and list(sample.columns) == ["text_raw", "text_clean", "label"]
    assert len(sample) == meta["dataset"]["n_test"]
    assert sample.text_clean.tolist() == [clean_text_v2(t) for t in sample.text_raw]
    assert all(" url" in t for t in sample.text_clean)  # ссылки КАПСОМ замаскированы

    with TestClient(create_app(make_settings(out / "model.joblib", out / "model_meta.json"))) as client:
        assert client.get("/healthcheck").json()["model_version"] == "tfidf-logreg@t"
        assert client.get("/model-info").json()["config"]["normalizer"] == NORMALIZER_V2
        assert client.post("/predict", json={"text": "Скидка СМОТРИ HTTP://Y.RU"}).status_code == 200
