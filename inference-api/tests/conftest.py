"""Общие данные и фикстуры тестов: игрушечные LR и ансамбль, фабрика Settings."""

from pathlib import Path

import numpy as np
import pytest
import torch

from app.config import Settings
from app.ensemble import CharCNN, Encoding, EnsemblePredictor, SpamClickbaitNet, build_vocabularies, save_bundle
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


@pytest.fixture(scope="session")
def toy_texts() -> list[str]:
    return [clean_text(text) for text, _ in TOY_DATA]


@pytest.fixture(scope="session")
def toy_labels() -> np.ndarray:
    return np.array([label for _, label in TOY_DATA])


@pytest.fixture(scope="session")
def toy_lr(toy_texts, toy_labels):
    """Маленькая TF-IDF + LR: обучается за доли секунды, настоящие артефакты не нужны."""
    cfg = {**BASE_CONFIG, "word_min_df": 1, "char_min_df": 1}
    return build_pipeline(cfg, toy_labels).fit(toy_texts, toy_labels)


@pytest.fixture(scope="session")
def make_settings():
    """Фабрика Settings с порогами по умолчанию; без meta_path — несуществующий файл рядом с моделью."""

    def make(model_path: Path, meta_path: Path | None = None) -> Settings:
        return Settings(model_path=model_path, meta_path=meta_path or model_path.with_name("unused.json"))

    return make


@pytest.fixture(scope="session")
def ensemble_bundle(tmp_path_factory, toy_texts, toy_lr):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    encoding = Encoding(max_len=16, max_chars=24, vocab_limit=100, min_freq=1)
    vocab, char_vocab = build_vocabularies(toy_texts, encoding)
    predictor = EnsemblePredictor(
        SpamClickbaitNet(len(vocab), len(char_vocab)),
        CharCNN(len(char_vocab)),
        toy_lr,
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
