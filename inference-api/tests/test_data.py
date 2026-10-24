"""Чтение df.csv и подключение Datasets/scripts."""

import sys

from training import train
from training.data import deduplicate, import_dataset_scripts, load_fixed_csv


def test_load_fixed_csv_skips_bad_lines(tmp_path):
    path = tmp_path / "df.csv"
    path.write_text(
        'text,is_spam1_or_clickbait2\n"привет, как дела",0\nкупи; срочно;1\n'
        "без разделителя\nметка,x\nчужой класс,7\n<<<<<<< HEAD\n\nшок,2\n",
        encoding="utf-8-sig",
    )
    texts, labels, bad = load_fixed_csv(path)
    assert texts == ["привет, как дела", "купи; срочно", "шок"]
    assert labels == [0, 1, 2]
    assert dict(bad) == {"нет разделителя": 2, "метка не число": 1, "метка вне {0,1,2}": 1}


def test_deduplicate_drops_duplicates_and_conflicts():
    texts, labels, stats = deduplicate(["a", "a", "b", "b", "c"], [0, 0, 1, 2, 1])
    assert (texts, labels) == (["a", "c"], [0, 1])
    assert stats == {"duplicates_removed": 1, "conflicting_texts_removed": 1}


def test_train_reexports_data_helpers():
    assert train.load_fixed_csv is load_fixed_csv and train.deduplicate is deduplicate


def test_import_dataset_scripts_adds_path_once(tmp_path, monkeypatch):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "dataset_filtering.py").write_text("def filter_dataset(frame):\n    return 'filtered'\n")
    (scripts / "dataset_splitting.py").write_text("def make_data_splits(frame, clean):\n    return 'split'\n")
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in ("dataset_filtering", "dataset_splitting"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    for _ in range(2):
        filter_dataset, make_data_splits = import_dataset_scripts(tmp_path / "df.csv")
    assert sys.path.count(str(scripts)) == 1
    assert (filter_dataset(None), make_data_splits(None, None)) == ("filtered", "split")
    for name in ("dataset_filtering", "dataset_splitting"):
        sys.modules.pop(name, None)
