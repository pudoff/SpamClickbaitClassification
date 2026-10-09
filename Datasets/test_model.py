"""
pytest-тесты для Модели 2 (Word + Char N-grams) проекта «Классификация спама и кликбейта».

Запуск:      pytest test_model.py -v
Артефакты:   model.pkl        - обученный пайплайн Модели 2;
             test_sample.pkl  - размеченная тестовая выборка (91110 сообщений).
"""

import pickle
import random
import re

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


URL_RE = re.compile(r"https?://\S+|www\.\S+|<link>")
HTML_RE = re.compile(r"<[^>]+>")
MENTION_RE = re.compile(r"[@#]\w+")
DIGIT_RE = re.compile(r"\d+([.,]\d+)?")
SPACE_RE = re.compile(r"\s+")


def clean_text(text: str, fix_yo: bool = True) -> str:
    """Нормализация сырого текста."""
    text = str(text)
    text = HTML_RE.sub(" ", text)
    text = URL_RE.sub(" URL ", text)
    text = MENTION_RE.sub(" ", text)
    text = text.lower()
    if fix_yo:
        text = text.replace("ё", "е")
    text = DIGIT_RE.sub(" NUM ", text)
    text = re.sub(r"[^a-zA-Zа-яА-Я\s]", " ", text)
    return SPACE_RE.sub(" ", text).strip()


CLASS_NAMES = {0: "обычное", 1: "спам", 2: "кликбейт"}

# Латиница-гомоглифы: визуально неотличимы от кириллицы, но для char n-грамм - другие символы
HOMOGLYPHS = {
    "а": "a", "е": "e", "о": "o", "с": "c", "р": "p",
    "х": "x", "у": "y", "к": "k", "м": "m", "т": "t",
}

# Загрузка артефактов.

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("test_sample.pkl", "rb") as f:
    SAMPLE: pd.DataFrame = pickle.load(f)


@pytest.fixture(scope="module")
def predictions():
    """Предсказания модели на всей тестовой выборке. Считаются один раз на модуль."""
    cleaned = [clean_text(t) for t in SAMPLE["text_raw"]]
    return model.predict(cleaned), model.predict_proba(cleaned)

# Тест 1. Общая метрика.

def test_overall_quality_meets_threshold(predictions):
    """Модель 2 должна держать планку качества на реальной размеченной выборке.
    Ориентиры из MLflow (обновлённый датасет): лучшая конфигурация п.12.4
    accuracy 0.8517 / macro-F1 0.8134, контрольная точка п.8 - 0.8069 / 0.7805.
    Пороги 0.80 / 0.78 рассчитаны так, чтобы проходить в обоих случаях.
    """
    y_true = SAMPLE["label"].values
    y_pred = predictions[0]

    # Cинхронность очистки: локальная clean_text обязана совпадать с обучающей.
    probe = SAMPLE["text_raw"].head(500)
    drifted = sum(
        clean_text(raw) != trained
        for raw, trained in zip(probe, SAMPLE["text_clean"].head(500))
    )
    assert drifted == 0, (
        f"clean_text в тестах разошлась с обучением на {drifted} примерах из 500 - "
        "метрики ниже нельзя интерпретировать"
    )

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    print(f"\n[test 1] n={len(y_true)}, accuracy={accuracy:.4f}, f1_macro={f1_macro:.4f}")

    assert accuracy >= 0.80, f"accuracy {accuracy:.4f} ниже порога 0.80 (ожидалось ~0.85)"
    assert f1_macro >= 0.78, f"macro-F1 {f1_macro:.4f} ниже порога 0.78 (ожидалось ~0.81)"


# Тест 2. Конкретное ожидаемое поведение: пол качества на целевом классе.

def test_clickbait_quality_floor(predictions):
    """Кликбейт - целевой класс проекта и самый слабый.
    Фиксируем для него явный пол качества, чтобы регрессия по минорному классу
    не могла пройти незамеченной под общей точностью.
    """
    y_true = SAMPLE["label"].values
    y_pred = predictions[0]

    f1_cb = f1_score(y_true, y_pred, labels=[2], average=None)[0]
    recall_cb = recall_score(y_true, y_pred, labels=[2], average=None)[0]
    precision_cb = precision_score(y_true, y_pred, labels=[2], average=None)[0]
    print(
        f"\n[test 2] Кликбейт: precision={precision_cb:.4f}, "
        f"recall={recall_cb:.4f}, f1={f1_cb:.4f}"
    )

    # Ориентиры п.12.4: F1 кликбейта 0.6499, recall 0.661.
    # Пороги занижены с запасом; при class_weight="balanced" recall был бы 0.765,
    # но precision 0.521 - поэтому проверяем именно сбалансированный минимум.
    assert f1_cb >= 0.60, f"F1 кликбейта {f1_cb:.4f} упал ниже порога 0.60"
    assert recall_cb >= 0.63, f"recall кликбейта {recall_cb:.4f} ниже порога 0.63"


# Тест 3. Устойчивость к опечаткам и искажениям.

def _distort(text: str, mode: str, rng: random.Random):
    """Вносит реалистичное искажение в одно слово длиннее 5 символов."""
    words = text.split()
    candidates = [i for i, w in enumerate(words) if len(w) >= 5]
    if not candidates:
        return None
    i = rng.choice(candidates)
    word = list(words[i])

    if mode == "swap":                      # перестановка соседних букв
        j = len(word) // 2
        word[j], word[j + 1] = word[j + 1], word[j]
    elif mode == "drop":                    # пропущенная буква
        word.pop(len(word) // 2)
    elif mode == "latin":                   # русская буква заменена латинским гомоглифом
        positions = [k for k, ch in enumerate(word) if ch in HOMOGLYPHS]
        if not positions:
            return None
        k = rng.choice(positions)
        word[k] = HOMOGLYPHS[word[k]]
    else:
        raise ValueError(mode)

    words[i] = "".join(word)
    return " ".join(words)


def test_robustness_to_typos_and_distortions():
    """Зачем модели char_wb n-граммы: они должны гасить опечатки.
    Для каждого вида искажения проверяем, что предсказание сохраняется.
    Порог 0.90, а не 1.0: при грубых искажениях часть сообщений действительно
    меняет класс, но не меньше 90% должно уцелеть.
    """
    rng = random.Random(42)
    cleaned = [clean_text(t) for t in SAMPLE["text_raw"]]
    indices = rng.sample(range(len(cleaned)), 400)

    report = {}
    for mode in ("swap", "drop", "latin"):
        base, distorted = [], []
        for j in indices:
            bad = _distort(cleaned[j], mode, rng)
            if bad is None:
                continue
            base.append(cleaned[j])
            distorted.append(bad)

        stability = float((model.predict(base) == model.predict(distorted)).mean())
        report[mode] = (stability, len(base))

    for mode, (stability, n) in report.items():
        print(f"\n[test 3] искажение '{mode}': устойчивость {stability:.4f} на {n} примерах")

    for mode, (stability, n) in report.items():
        assert stability >= 0.90, (
            f"при искажении '{mode}' предсказание изменилось у {(1 - stability):.1%} "
            f"сообщений из {n} - char n-граммы не гасят этот тип искажений"
        )

# Тест 4. Устойчивость к шуму, который удаляется при очистке.

def test_invariance_to_noise_removed_by_cleaning():
    """Проверяем два разных вида шума.

    1) Шум, который clean_text удаляет по построению (капс, HTML-теги, лишние
       пробелы, эмодзи, пунктуация): очищенный текст обязан совпасть побайтно,
       и предсказание - тоже.
    2) Ссылки и @/#-упоминания: clean_text не удаляет их, а превращает в токен
       «URL», поэтому они становятся признаками TF-IDF. Здесь требуем устойчивости
       не 100%, а >= 90%; фактическая чувствительность печатается и разбирается
       в выводах как найденное слабое место.
    """
    # Сравниваем результат ОДНОГО и того же конвейера на сыром тексте и на зашумлённом.
    # Нельзя сравнивать clean_text(clean_text(x)) с clean_text(x): clean_text не идемпотентна,
    # т.к. токен-маска чисел " NUM " вставляется уже после lower() и остаётся заглавным.
    raw = list(SAMPLE["text_raw"].head(300))
    base_clean = [clean_text(t) for t in raw]

    def add_removable_noise(text: str) -> str:
        return "  " + text.upper() + "   <b> </i> !!! \U0001F52A \U0001F4A5 ...  "

    def add_token_noise(text: str) -> str:
        return text + " http://example.com/promo www.example.org @nickname #hashtag"

    # --- 1) полностью удаляемый шум
    removable_clean = [clean_text(add_removable_noise(t)) for t in raw]
    text_equal = sum(a == b for a, b in zip(removable_clean, base_clean))
    pred_equal = float((model.predict(removable_clean) == model.predict(base_clean)).mean())

    print(
        f"\n[test 4a] удаляемый шум: текст совпал {text_equal}/{len(raw)}, "
        f"предсказания совпали {pred_equal:.4f}"
    )

    assert text_equal == len(raw), (
        f"clean_text зависит от регистра: {len(raw) - text_equal} из {len(raw)} текстов "
        "изменились после вставки шума, хотя шум удаляется по построению. "
        "Причина: URL_RE / MENTION_RE / HTML_RE применяются ДО text.lower(), поэтому в "
        "сообщениях, набранных КАПСОМ (типичный стиль спама), ссылки и @/#-упоминания "
        "не маскируются и попадают в признаки как обычные слова. "
        "Исправление: перенести text.lower() в самое начало clean_text и переобучить модель."
    )
    assert pred_equal == 1.0, (
        f"на {1 - pred_equal:.1%} сообщений шум изменил предсказание, хотя очищенный "
        "текст совпал побайтно - модель недетерминирована"
    )

    # --- 2) шум, превращаемый в признаки (ссылки, упоминания, хештеги)
    token_clean = [clean_text(add_token_noise(t)) for t in raw]
    token_stability = float((model.predict(token_clean) == model.predict(base_clean)).mean())

    print(
        f"[test 4b] ссылки и @/#-упоминания: устойчивость {token_stability:.4f} "
        f"(предсказание изменилось у {1 - token_stability:.1%} сообщений)"
    )

    assert token_stability >= 0.90, (
        f"ссылки и @/#-упоминания изменили предсказание у {1 - token_stability:.1%} сообщений "
        f"(устойчивость {token_stability:.4f} < 0.90) - текст можно перевести в другой класс "
        "простым добавлением ссылки"
    )

# Тест 5. Крайний случай: очень длинный текст.

def test_very_long_text_does_not_break_model():
    """Длинные сообщения не должны ронять модель, ломать распределение вероятностей
    и не должны «выбивать» текст из его класса из-за насыщения TF-IDF."""
    cleaned = [clean_text(t) for t in SAMPLE["text_raw"]]
    labels = SAMPLE["label"].values

    clickbait_seed = next(c for c, lb in zip(cleaned, labels) if lb == 2)
    spam_seed = next(c for c, lb in zip(cleaned, labels) if lb == 1)

    long_clickbait = " ".join([clickbait_seed] * 400)
    long_spam = " ".join([spam_seed] * 400)
    long_mixed = " ".join(cleaned[:50] * 80)

    texts = [long_clickbait, long_spam, long_mixed]
    preds = model.predict(texts)
    probas = model.predict_proba(texts)

    print(
        f"\n[test 5] длина {len(long_clickbait.split())} слов; предсказания: "
        f"кликбейт*400 -> {CLASS_NAMES[preds[0]]}, спам*400 -> {CLASS_NAMES[preds[1]]}, "
        f"смешанный -> {CLASS_NAMES[preds[2]]}"
    )
    print(f"[test 5] суммы вероятностей: {np.round(probas.sum(axis=1), 6).tolist()}")

    assert all(p in CLASS_NAMES for p in preds), f"модель вернула класс вне {CLASS_NAMES}"
    assert np.allclose(probas.sum(axis=1), 1.0), (
        f"сумма вероятностей не равна 1: {probas.sum(axis=1)}"
    )
    assert preds[0] == 2, (
        f"кликбейт, повторённый 400 раз, классифицирован как «{CLASS_NAMES[preds[0]]}» - "
        "насыщение TF-IDF ломает длинные тексты"
    )
    assert preds[1] == 1, (
        f"спам, повторённый 400 раз, классифицирован как «{CLASS_NAMES[preds[1]]}» - "
        "насыщение TF-IDF ломает длинные тексты"
    )

# Тест 6. Подгруппа данных: короткие сообщения.

def test_short_messages_subgroup_quality(predictions):
    """Отдельный режим работы: короткие реплики (<= 3 слов).

    Ожидаем заметно худшее качество, чем на среднем по выборке: контекста мало,
    а TF-IDF-признаки коротких сообщений бедны. Именно поэтому здесь смотрим
    на macro-F1, а не только на accuracy: accuracy на этой подгруппе завышена
    из-за преобладания класса «Обычные».

    Абсолютный порог здесь бессмыслен: подгруппа на 70 % состоит из класса
    "Обычные", поэтому accuracy завышена. Используем относительный критерий:
    macro-F1 коротких сообщений должен быть не ниже 85 % от macro-F1 по всей
    выборке. Факт на новых данных: 0.53 против 0.81 - критерий не выполняется.
    """
    y_true = SAMPLE["label"].values
    y_pred = predictions[0]
    n_words = SAMPLE["n_words"].values

    mask = n_words <= 3
    assert mask.sum() > 100, "в выборке слишком мало коротких сообщений для оценки"

    y_short, pred_short = y_true[mask], y_pred[mask]
    accuracy = accuracy_score(y_short, pred_short)
    f1_macro = f1_score(y_short, pred_short, average="macro")
    per_class = f1_score(y_short, pred_short, labels=[0, 1, 2], average=None)
    dist = np.bincount(y_short, minlength=3).tolist()

    print(
        f"\n[test 6] коротких сообщений: {mask.sum()} ({mask.mean():.1%} выборки), "
        f"accuracy={accuracy:.4f}, f1_macro={f1_macro:.4f}"
    )
    print(f"[test 6] F1 по классам: {[round(v, 4) for v in per_class]}, распределение: {dist}")
    print(
        f"[test 6] для сравнения, вся выборка: "
        f"accuracy={accuracy_score(y_true, y_pred):.4f}, "
        f"f1_macro={f1_score(y_true, y_pred, average='macro'):.4f}"
    )

    overall_f1 = f1_score(y_true, y_pred, average="macro")
    assert accuracy >= 0.55, (
        f"accuracy на коротких сообщениях {accuracy:.4f} ниже порога 0.55 - "
        "модель почти не работает на коротких репликах"
    )
    required = 0.85 * overall_f1
    assert f1_macro >= required, (
        f"macro-F1 на коротких сообщениях {f1_macro:.4f} - это "
        f"{f1_macro / overall_f1:.0%} от общего macro-F1 {overall_f1:.4f}, "
        f"а требуется не ниже {required:.4f} (85 %). Минорные классы проседают "
        "на коротких репликах сильнее всего."
    )
