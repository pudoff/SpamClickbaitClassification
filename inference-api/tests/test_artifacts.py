"""Общие функции артефактов: контрольная сумма и загрузка LR."""

import hashlib
import pickle

import joblib

from app import ensemble
from app.artifacts import load_artifact, sha256_of


def test_sha256_matches_hashlib_and_old_name(tmp_path):
    path = tmp_path / "blob.bin"
    data = bytes(range(256)) * 5000
    path.write_bytes(data)
    assert sha256_of(path) == sha256_of(str(path)) == hashlib.sha256(data).hexdigest()
    assert ensemble.file_hash is sha256_of


def test_load_artifact_reads_joblib_and_pickle(tmp_path):
    value = {"classes": [0, 1, 2]}
    joblib.dump(value, tmp_path / "model.joblib")
    with (tmp_path / "model.pkl").open("wb") as f:
        pickle.dump(value, f)
    assert load_artifact(tmp_path / "model.joblib") == load_artifact(tmp_path / "model.pkl") == value
