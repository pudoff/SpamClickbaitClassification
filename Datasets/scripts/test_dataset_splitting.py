"""Проверки независимости частей и групповой кросс-валидации."""

import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from dataset_splitting import make_data_splits, save_split_manifest
from sklearn.model_selection import StratifiedGroupKFold


def fixture():
    rows = []
    for label in (0, 1, 2):
        for number in range(60):
            text = f"Текст категории {label} образец {number}"
            rows.extend([(text, label), (text.upper() + "!!!", label)])
    return pd.DataFrame(rows, columns=["text", "is_spam1_or_clickbait2"])


def normalize(text):
    return re.sub(r"[^\w\s]", "", text.casefold()).strip()


class DataSplittingTests(unittest.TestCase):
    def test_all_rows_once_and_no_group_overlap(self):
        frame = fixture()
        split = make_data_splits(frame, normalize)
        all_indices = np.concatenate([split[k] for k in ("fit", "dev", "test")])
        self.assertEqual(sorted(all_indices), list(range(len(frame))))
        self.assertEqual(len(set(all_indices)), len(frame))
        groups = {k: set(split["keys"].iloc[split[k]]) for k in ("fit", "dev", "test")}
        for a, b in (("fit", "dev"), ("fit", "test"), ("dev", "test")):
            self.assertFalse(groups[a] & groups[b])
        for idx in split.values():
            if isinstance(idx, np.ndarray):
                self.assertEqual(set(frame.iloc[idx].is_spam1_or_clickbait2), {0, 1, 2})

    def test_deterministic_partitions_and_fingerprint(self):
        frame = fixture()
        a, b = make_data_splits(frame, normalize), make_data_splits(frame, normalize)
        for key in ("fit", "dev", "test"):
            np.testing.assert_array_equal(a[key], b[key])
        self.assertEqual(a["summary"]["split_id"], b["summary"]["split_id"])
        changed = make_data_splits(frame, normalize, random_state=73)
        self.assertNotEqual(a["summary"]["split_id"], changed["summary"]["split_id"])

    def test_conflicting_labels_do_not_split_identical_text(self):
        frame = fixture()
        frame.loc[1, "is_spam1_or_clickbait2"] = 2
        split = make_data_splits(frame, normalize)
        self.assertEqual(split["summary"]["groups_with_conflicting_labels"], 1)
        self.assertTrue(any({0, 1}.issubset(set(split[k])) for k in ("fit", "dev", "test")))

    def test_target_sizes_on_uniform_groups(self):
        split = make_data_splits(fixture(), normalize)
        self.assertEqual(len(split["test"]), 72)
        self.assertEqual(len(split["dev"]), 30)
        self.assertEqual(len(split["fit"]), 258)

    def test_cv_never_splits_training_group(self):
        frame = fixture()
        split = make_data_splits(frame, normalize)
        idx = split["fit"]
        groups = split["keys"].iloc[idx].to_numpy()
        y = frame.iloc[idx].is_spam1_or_clickbait2.to_numpy()
        cv = StratifiedGroupKFold(3, shuffle=True, random_state=42)
        for fit, validation in cv.split(np.zeros(len(idx)), y, groups):
            self.assertFalse(set(groups[fit]) & set(groups[validation]))

    def test_manifest_preserves_membership_and_hashes_groups(self):
        frame = fixture()
        split = make_data_splits(frame, normalize)
        with tempfile.TemporaryDirectory() as directory:
            save_split_manifest(frame, split, directory)
            saved = pd.read_csv(Path(directory) / "manifest.csv")
            for name in ("fit", "dev", "test"):
                self.assertEqual(set(saved.loc[saved.partition.eq(name), "prepared_row"]), set(split[name]))
            self.assertEqual(saved.normalized_group_hash.nunique(), 180)
            summary = json.loads((Path(directory) / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary, split["summary"])

    def test_invalid_labels_and_sizes_rejected(self):
        frame = fixture()
        frame.loc[0, "is_spam1_or_clickbait2"] = 2000
        with self.assertRaises(ValueError):
            make_data_splits(frame, normalize)
        with self.assertRaises(ValueError):
            make_data_splits(fixture(), normalize, test_size=1)


if __name__ == "__main__":
    unittest.main()
