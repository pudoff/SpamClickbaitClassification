"""Проверка границ новых правил: python -m unittest discover -s Datasets/scripts -p test_dataset_filtering.py."""

import unittest

import pandas as pd
from dataset_filtering import filter_dataset, inspect_text


class DatasetFilteringTests(unittest.TestCase):
    def test_symbols_numbers_and_latin_do_not_pass(self):
        for text in (". . . . .", "* * * *", "/ 1 / 1 / 1 / 1", "one two three"):
            with self.subTest(text=text):
                self.assertIn("no_cyrillic", inspect_text(text)[3])

    def test_mixed_text_with_one_russian_word_is_excluded(self):
        cleaned, count, _, reasons = inspect_text("голова ijsri nwz jpd")
        self.assertEqual(count, 1)
        self.assertEqual(reasons, ["fewer_than_3_cyrillic_words"])

    def test_three_words_with_uppercase_yo_and_punctuation(self):
        self.assertEqual(inspect_text("Ёжик, ЁЛКА и!")[1:4], (3, [], []))

    def test_emojis_around_words_do_not_make_text_too_short(self):
        self.assertEqual(inspect_text("Привет🙂 как дела")[1:4], (3, [], []))

    def test_mixed_alphanumeric_tokens_do_not_count_as_russian_words(self):
        self.assertEqual(inspect_text("голова русenglish слово123")[1], 1)

    def test_hyphenated_word_is_one_whitespace_word(self):
        self.assertEqual(inspect_text("кое-кто принёс подарок")[1], 3)

    def test_three_repeats_remain(self):
        text = "Привет привет привет"
        self.assertEqual(inspect_text(text), (text, 3, [], []))

    def test_four_repeats_removed_including_single_latin_letters(self):
        cleaned, count, fragments, reasons = inspect_text("Хорошее начало этого сообщения d d d d")
        self.assertEqual(cleaned, "Хорошее начало этого сообщения")
        self.assertEqual((count, fragments, reasons), (4, ["d d d d"], []))

    def test_runs_in_middle_removed_without_losing_good_end(self):
        cleaned, count, fragments, reasons = inspect_text(
            "Сегодня was was was was состоится важная встреча wit wit wit wit"
        )
        self.assertEqual(cleaned, "Сегодня состоится важная встреча")
        self.assertEqual((count, len(fragments), reasons), (4, 2, []))

    def test_punctuation_case_tabs_and_newlines_in_repeats(self):
        cleaned, _, fragments, _ = inspect_text("Мы обсуждаем важные дела Ёж,\tёж\nЕЖ! еж.")
        self.assertEqual(cleaned, "Мы обсуждаем важные дела")
        self.assertEqual(len(fragments), 1)

    def test_punctuation_does_not_join_nonconsecutive_repeats(self):
        text = "Мы обсуждаем важные дела d d ! d d"
        self.assertEqual(inspect_text(text)[0], text)

    def test_filter_applies_minimum_after_removing_fragment(self):
        cleaned, count, fragments, reasons = inspect_text(".. .. ..найти хитрые хитрые хитрые хитрые хитрости ... ..")
        self.assertEqual(count, 2)
        self.assertTrue(fragments)
        self.assertEqual(reasons, ["fewer_than_3_cyrillic_words"])

    def test_missing_value_is_excluded(self):
        for value in (None, float("nan")):
            self.assertIn("no_cyrillic", inspect_text(value)[3])

    def test_removed_garbage_does_not_create_new_unhandled_repeats(self):
        text = "Сегодня состоится важная встреча да да d d d d да да"
        cleaned, count, fragments, reasons = inspect_text(text)
        self.assertEqual(cleaned, "Сегодня состоится важная встреча")
        self.assertEqual((count, fragments, reasons), (4, ["d d d d", "да да да да"], []))
        self.assertEqual(inspect_text(cleaned), (cleaned, 4, [], []))

    def test_journal_source_lines_labels_and_input_not_modified(self):
        frame = pd.DataFrame(
            {
                "text": ["Три обычных слова d d d d", "голова js", "Это три слова"],
                "is_spam1_or_clickbait2": [0, 1, 2000],
            }
        )
        frame.attrs["source_lines"] = [569398, 569406, 9]
        before = frame.copy(deep=True)
        cleaned, journal, stats = filter_dataset(frame)
        pd.testing.assert_frame_equal(frame, before)
        self.assertEqual(cleaned.text.tolist(), ["Три обычных слова"])
        self.assertEqual(cleaned.attrs["source_lines"], [569398])
        self.assertEqual(journal.action.tolist(), ["corrected", "excluded", "excluded"])
        self.assertEqual(journal.source_line.tolist(), [569398, 569406, 9])
        self.assertEqual(stats["excluded_rows"], 2)
        self.assertEqual(stats["invalid_label"], 1)

    def test_empty_frame_returns_empty_journal_and_zero_counts(self):
        cleaned, journal, stats = filter_dataset(pd.DataFrame(columns=["text", "is_spam1_or_clickbait2"]))
        self.assertTrue(cleaned.empty)
        self.assertTrue(journal.empty)
        self.assertEqual(stats["excluded_rows"], 0)


if __name__ == "__main__":
    unittest.main()
