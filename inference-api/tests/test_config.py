"""Settings: значения по умолчанию и разбор переменных окружения."""

from pathlib import Path

import pytest
import torch

from app.config import DEFAULT_TAU, DEFAULT_TAU_SPAM, Settings
from app.ensemble import load_ensemble

ENV = ("MODEL_PATH", "MODEL_META_PATH", "TAU", "TAU_SPAM", "TORCH_NUM_THREADS")


@pytest.fixture
def clean_env(monkeypatch):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_defaults(clean_env):
    settings = Settings.from_env()
    assert settings.model_path == Path("models/ensemble/manifest.json")
    assert settings.meta_path == Path("models/ensemble/model_meta.json")
    assert (settings.tau, settings.tau_spam, settings.torch_num_threads) == (0.55, 0.65, 2)
    assert (DEFAULT_TAU, DEFAULT_TAU_SPAM) == (0.55, 0.65)
    assert Settings(model_path=Path("m"), meta_path=Path("x")) == Settings(Path("m"), Path("x"), 0.55, 0.65, 2)


def test_env_overrides(clean_env):
    clean_env.setenv("MODEL_PATH", "/models/model.joblib")
    clean_env.setenv("TAU", "0.4")
    clean_env.setenv("TAU_SPAM", "0.9")
    clean_env.setenv("TORCH_NUM_THREADS", "3")
    settings = Settings.from_env()
    assert settings.meta_path == Path("/models/model_meta.json")
    assert (settings.tau, settings.tau_spam, settings.torch_num_threads) == (0.4, 0.9, 3)


@pytest.mark.parametrize(("name", "value"), [("TAU", "1.5"), ("TAU_SPAM", "-0.1"), ("TORCH_NUM_THREADS", "0")])
def test_invalid_env_rejected(clean_env, name, value):
    clean_env.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        Settings.from_env()


def test_load_ensemble_threads(ensemble_bundle):
    path, _ = ensemble_bundle
    load_ensemble(path, num_threads=1)
    assert torch.get_num_threads() == 1
    load_ensemble(path)
    assert torch.get_num_threads() == 2
    with pytest.raises(ValueError):
        load_ensemble(path, num_threads=0)
