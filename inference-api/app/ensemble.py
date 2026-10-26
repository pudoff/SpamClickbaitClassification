"""Инференс замороженных моделей из 2_RNN_CNN_LR.ipynb.

Порядок токенов, усечение, поверхностные признаки и слои сетей намеренно совпадают
с обученным ноутбуком. Их изменение требует нового экспорта и проверки совпадения.
"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import sklearn
import torch
from torch import nn
from torch.nn import functional as F

from app.artifacts import sha256_of
from app.config import DEFAULT_TORCH_THREADS
from app.labels import CLASS_IDS, N_CLASSES

WORD_RE = re.compile(r"[a-zа-я]+(?:'[a-zа-я]+)?")
CYR = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
FORMAT = "spam-clickbait-ensemble-v1"
N_SURF = 12
# Компоненты смеси (порядок суммирования) и файлы комплекта: менять только с новым форматом.
COMPONENTS = ("rnn", "char", "lr")
BUNDLE_FILES = {"rnn": "rnn.pt", "char": "char.pt", "lr": "lr.joblib", "vocab": "vocab.json"}


@dataclass(frozen=True)
class Encoding:
    max_len: int = 256
    max_chars: int = 256
    vocab_limit: int = 60000
    min_freq: int = 2


DEFAULT_ENCODING = Encoding()


def word_ngrams(text):
    words = WORD_RE.findall(text)
    return words + [words[i] + "_" + words[i + 1] for i in range(len(words) - 1)]


def build_vocabularies(texts, encoding=DEFAULT_ENCODING):
    """Повторяет устойчивый порядок по частоте из ноутбука; только по текстам fit."""
    counter = {}
    for text in texts:
        for token in word_ngrams(text):
            counter[token] = counter.get(token, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, count in sorted(counter.items(), key=lambda item: -item[1]):
        if count >= encoding.min_freq and len(vocab) < encoding.vocab_limit:
            vocab[token] = len(vocab)
    char_freq = {}
    for text in texts[:100000]:
        for char in text[: encoding.max_chars]:
            char_freq[char] = char_freq.get(char, 0) + 1
    char_vocab = {"<pad>": 0, "<unk>": 1}
    for char, _ in sorted(char_freq.items(), key=lambda item: -item[1]):
        if len(char_vocab) < 500:
            char_vocab[char] = len(char_vocab)
    return vocab, char_vocab


def encode_texts(texts, vocab, char_vocab, encoding=DEFAULT_ENCODING):
    word_ids = np.zeros((len(texts), encoding.max_len), dtype=np.int64)
    char_ids = np.zeros((len(texts), encoding.max_chars), dtype=np.int64)
    surf = np.zeros((len(texts), N_SURF), dtype=np.float32)
    for i, text in enumerate(texts):
        tokens = word_ngrams(text)[: encoding.max_len]
        word_ids[i, : len(tokens)] = [vocab.get(token, 1) for token in tokens]
        chars = text[: encoding.max_chars]
        char_ids[i, : len(chars)] = [char_vocab.get(char, 1) for char in chars]
        nchar = max(len(text), 1)
        letters = [char for char in text if char.isalpha()] or ["a"]
        nl = len(letters)
        words = text.split() or ["a"]
        lengths = [len(word) for word in words]
        surf[i] = [
            np.log1p(len(text)),
            np.log1p(len(words)),
            np.mean(lengths),
            max(lengths),
            sum(char in CYR for char in letters) / nl,
            sum(char.isascii() and char.isalpha() for char in letters) / nl,
            sum(char.isdigit() for char in text) / nchar,
            sum(char.isupper() for char in letters) / nl,
            float(" url " in text),
            len(set(words)) / len(words),
            np.log1p(text.count(" ")),
            text.count("!"),
        ]
    return word_ids, char_ids, surf


class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.score = nn.Linear(dim, 1)

    def forward(self, hidden, mask):
        scores = self.score(torch.tanh(self.proj(hidden))).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)
        return context, weights


class SpamClickbaitNet(nn.Module):
    def __init__(
        self, vocab_size, char_vocab_size, emb_dim=128, hidden=128, char_channels=128, use_char=True, dropout=0.4
    ):
        super().__init__()
        self.use_char = use_char
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.attention = AttentionPool(hidden * 2)
        d_char = 0
        if use_char:
            self.char_emb = nn.Embedding(char_vocab_size, 32, padding_idx=0)
            self.char_conv1 = nn.Conv1d(32, char_channels, 5, padding=2)
            self.char_bn1 = nn.BatchNorm1d(char_channels)
            self.char_conv2 = nn.Conv1d(char_channels, char_channels, 5, padding=2)
            self.char_bn2 = nn.BatchNorm1d(char_channels)
            d_char = char_channels
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + d_char + N_SURF, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(64, 3)

    def forward(self, word_ids, char_ids, surf, return_attention=False):
        mask = word_ids != 0
        emb = self.embedding(word_ids) * mask.unsqueeze(-1).float()
        seq_out, _ = self.gru(emb)
        word_ctx, attn = self.attention(seq_out, mask)
        parts = [word_ctx, surf]
        if self.use_char:
            chars = self.char_emb(char_ids).transpose(1, 2)
            chars = F.relu(self.char_bn1(self.char_conv1(chars)))
            chars = F.relu(self.char_bn2(self.char_conv2(chars)))
            parts.append(chars.max(dim=2).values)
        logits = self.classifier(self.head(torch.cat(parts, dim=1)))
        return (logits, attn) if return_attention else logits


class CharCNN(nn.Module):
    def __init__(self, vocab_size, channels=128, n_classes=3, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.features = nn.Sequential(
            nn.Conv1d(64, channels, 5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, 5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, 5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(channels + N_SURF, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, char_ids, surf):
        chars = self.features(self.emb(char_ids).transpose(1, 2)).max(dim=2).values
        return self.head(torch.cat([chars, surf], dim=1))


# Старое имя: реэкспорт для внешних импортов.
file_hash = sha256_of


def blend(parts, weights, normalize=True):
    """Взвешенная смесь вероятностей компонентов; слагаемые суммируются в порядке COMPONENTS."""
    mixed = sum(weights[key] * parts[key] for key in COMPONENTS)
    return mixed / mixed.sum(axis=1, keepdims=True) if normalize else mixed


class EnsemblePredictor:
    classes_ = np.array(CLASS_IDS)

    def __init__(self, rnn, char, lr, vocab, char_vocab, weights, encoding=DEFAULT_ENCODING):
        values = np.asarray([weights[key] for key in COMPONENTS], dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or not np.isclose(values.sum(), 1):
            raise ValueError("веса ансамбля должны быть конечными, неотрицательными и в сумме давать 1")
        if list(lr.classes_) != list(CLASS_IDS):
            raise ValueError("классы LR должны идти в порядке [0, 1, 2]")
        self.rnn, self.char, self.lr = rnn.eval(), char.eval(), lr
        self.vocab, self.char_vocab = vocab, char_vocab
        self.weights, self.encoding = dict(weights), encoding

    @torch.inference_mode()
    def component_probabilities(self, cleaned_texts):
        words, chars, surf = encode_texts(cleaned_texts, self.vocab, self.char_vocab, self.encoding)
        words, chars, surf = map(torch.from_numpy, (words, chars, surf))
        return {
            "rnn": torch.softmax(self.rnn(words, chars, surf), dim=1).numpy(),
            "char": torch.softmax(self.char(chars, surf), dim=1).numpy(),
            "lr": self.lr.predict_proba(cleaned_texts),
        }

    def predict_proba(self, cleaned_texts):
        # Батчами: ограниченная память при проверке экспорта и для вызовов с большими списками.
        if not len(cleaned_texts):
            return np.empty((0, N_CLASSES))
        batches = []
        for start in range(0, len(cleaned_texts), 64):
            texts = cleaned_texts[start : start + 64]
            probabilities = blend(self.component_probabilities(texts), self.weights)
            if not np.isfinite(probabilities).all() or (probabilities < 0).any():
                raise ValueError("ансамбль вернул некорректное распределение вероятностей")
            batches.append(probabilities)
        return np.concatenate(batches)


def load_ensemble(manifest_path, *, num_threads=None):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT or manifest.get("classes") != list(CLASS_IDS):
        raise ValueError("неподдерживаемый формат комплекта или порядок классов")
    trained_sklearn = manifest["meta"].get("sklearn_version")
    if trained_sklearn and trained_sklearn != sklearn.__version__:
        raise ValueError(f"LR ансамбля требует scikit-learn {trained_sklearn}, установлен {sklearn.__version__}")
    paths = {}
    for key in BUNDLE_FILES:
        entry = manifest["files"][key]
        path = (manifest_path.parent / entry["path"]).resolve()
        if not path.is_relative_to(manifest_path.parent.resolve()):
            raise ValueError("файлы ансамбля должны лежать внутри каталога комплекта")
        if sha256_of(path) != entry["sha256"]:
            raise ValueError(f"не совпадает контрольная сумма: {key}")
        paths[key] = path
    threads = DEFAULT_TORCH_THREADS if num_threads is None else num_threads
    if threads < 1:
        raise ValueError("TORCH_NUM_THREADS должно быть не меньше 1")
    torch.set_num_threads(threads)
    vocab_data = json.loads(paths["vocab"].read_text(encoding="utf-8"))
    rnn = SpamClickbaitNet(len(vocab_data["word"]), len(vocab_data["char"]))
    char = CharCNN(len(vocab_data["char"]))
    rnn.load_state_dict(torch.load(paths["rnn"], map_location="cpu", weights_only=True))
    char.load_state_dict(torch.load(paths["char"], map_location="cpu", weights_only=True))
    predictor = EnsemblePredictor(
        rnn,
        char,
        joblib.load(paths["lr"]),
        vocab_data["word"],
        vocab_data["char"],
        manifest["weights"],
        Encoding(**manifest["encoding"]),
    )
    return predictor, manifest["meta"]


def save_bundle(folder, predictor, meta):
    """Сохраняет только то, что нужно для инференса: без оптимизатора и без глобальных объектов ноутбука."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(predictor.rnn.state_dict(), folder / BUNDLE_FILES["rnn"])
    torch.save(predictor.char.state_dict(), folder / BUNDLE_FILES["char"])
    joblib.dump(predictor.lr, folder / BUNDLE_FILES["lr"], compress=3)
    (folder / BUNDLE_FILES["vocab"]).write_text(
        json.dumps({"word": predictor.vocab, "char": predictor.char_vocab}, ensure_ascii=False), encoding="utf-8"
    )
    files = {key: {"path": name, "sha256": sha256_of(folder / name)} for key, name in BUNDLE_FILES.items()}
    manifest = {
        "format": FORMAT,
        "classes": list(CLASS_IDS),
        "weights": predictor.weights,
        "encoding": asdict(predictor.encoding),
        "files": files,
        "meta": meta,
    }
    path = folder / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
