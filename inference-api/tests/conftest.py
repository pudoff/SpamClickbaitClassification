"""Общие фикстуры тестов."""

import numpy as np
import pytest
import torch

from app.ensemble import CharCNN, Encoding, EnsemblePredictor, SpamClickbaitNet, build_vocabularies, save_bundle
from app.text import clean_text
from tests.test_api import TOY_DATA
from training.train import BASE_CONFIG, build_pipeline


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
