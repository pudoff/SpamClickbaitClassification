"""Тесты демонстрационного пайплайна (уровни: данные, сплит/утечки, модель, политика, дрейф).

Запуск: venv\\Scripts\\python.exe -m pytest FinalProject\\tests -q
Соответствие тест-плану: docs/TEST_PLAN.md (T-DATA-*, T-SPLIT-*, T-MODEL-*, T-POL-*, T-DRIFT-*).
"""

import numpy as np
import pytest

from demo import (
    build_pipeline,
    clean_text,
    decide,
    evaluate,
    group_split,
    ks_drift,
    load_dataset,
    policy_metrics,
    psi,
    template_key,
    validate_dataset,
)
from demo.drift import psi_level
from demo.models import blend_proba, select_weights

# ---------------------------------------------------------------- данные


def test_clean_text_masks_uppercase_url_and_numbers():
    """T-DATA-01: нормализация маскирует ссылки КАПСОМ и числа (проблема clean-text-v1)."""
    out = clean_text("СКИДКА 90% ПО ССЫЛКЕ HTTP://SPAM.EXE Ёлка")
    assert "http" not in out and "url" in out
    assert "90" not in out and "num" in out
    assert "ёлка" not in out and "елка" in out


def test_load_dataset_skips_dirty_rows(tmp_path):
    """T-DATA-02: загрузчик отбрасывает грязные строки и восстанавливает ';' записи."""
    p = tmp_path / "dirty.csv"
    p.write_text(
        "text,is_spam1_or_clickbait2\n"
        '"купи сейчас скидка акция магазин",1\n'
        '"купи сейчас скидка акция магазин",1\n'  # дубль
        '"привет как дела встретимся завтра",0\n'
        '"привет как дела встретимся завтра",2\n'  # конфликт меток -> обе удалить
        '"ты не поверишь что случилось дальше",2\n'
        '"возьми хлеба по дороге домой",0\n'
        '"текст с недопустимой меткой здесь",2000\n'
        '"текст с нечисловой меткой здесь",abc\n'
        '"запись через точку с запятой раз";1\n'
        "text,is_spam1_or_clickbait2\n"  # встроенный заголовок
        '"да",0\n',  # короткий текст (<3 слов)
        encoding="utf-8",
    )
    raw, clean, y, rep = load_dataset(p)
    assert len(clean) == 4
    assert rep.n_duplicates_removed == 1
    assert rep.n_conflict_texts_removed == 1
    assert rep.skipped["invalid_label"] == 1
    assert rep.skipped["no_integer_label"] == 1
    assert rep.skipped["embedded_header"] == 2  # первый заголовок + встроенный
    assert rep.n_short_removed == 1
    assert set(y.tolist()) == {0, 1, 2}


def test_validate_dataset_flags_conflicts_and_imbalance():
    """T-DATA-03: валидатор находит смешанные метки в шаблонных группах и дисбаланс."""
    # одна шаблонная группа (числа маскируются в num) со смешанными метками
    texts = [clean_text(f"скидка {i} процентов сегодня") for i in range(10)]
    labels = [1] * 5 + [0] * 5
    # много спама с уникальными текстами -> миноритарный класс < 10%
    texts += [clean_text(f"кредит одобрен без отказа заявка онлайн вариант{i}") for i in range(90)]
    labels += [1] * 90
    y = np.array(labels)
    report = validate_dataset(texts, y)
    assert report["n_conflict_groups"] >= 1
    assert any("смешанными метками" in s for s in report["issues"])
    assert any("дисбаланс" in s for s in report["issues"])


# ---------------------------------------------------------------- сплит


def test_group_split_no_leakage_and_full_coverage():
    """T-SPLIT-01: группы шаблонов не пересекаются между fit/dev/test; покрытие полное."""
    texts, y = [], []
    for i in range(60):
        base = f"шаблон номер num вариант {i % 10} слово{'абвгд'[i % 5]}"
        for num in (100, 200):  # два близких дубликата на шаблон
            texts.append(clean_text(base.replace("num", str(num))))
            y.append(i % 3)
    # уникальные очищенные тексты (дедупликация не нужна: тексты различаются)
    y = np.array(y)
    f, d, t = group_split(texts, y, seed=42)
    assert len(f) + len(d) + len(t) == len(texts)
    assert len(set(f) | set(d) | set(t)) == len(texts)
    kf = {template_key(texts[i]) for i in f}
    kd = {template_key(texts[i]) for i in d}
    kt = {template_key(texts[i]) for i in t}
    assert not (kf & kd) and not (kf & kt) and not (kd & kt)


def test_group_split_deterministic():
    """T-SPLIT-02: один seed — один и тот же сплит (воспроизводимость)."""
    texts = [clean_text(f"уникальный текст номер {i} про что-то") for i in range(50)]
    y = np.array([i % 3 for i in range(50)])
    a = group_split(texts, y, seed=42)
    b = group_split(texts, y, seed=42)
    for x, z in zip(a, b):
        assert np.array_equal(x, z)


# ---------------------------------------------------------------- модель


@pytest.fixture(scope="module")
def toy_corpus():
    rng = np.random.default_rng(0)
    spam = ["купи сейчас скидка акция магазин распродажа", "выигрыш приз получи деньги срочно звони", "кредит одобрен без отказа заявка онлайн"]
    click = ["ты не поверишь что случилось дальше шокирующая правда", "врачи скрывали это средство вот результат", "знаменитость показала фото все ахнули"]
    normal = ["привет как дела встретимся завтра в кафе", "возьми хлеба по дороге домой пожалуйста", "я закончил отчет отправлю вечером тебе"]
    texts, labels = [], []
    for lab, pool in enumerate([normal, spam, click]):
        for _ in range(40):
            base = pool[rng.integers(len(pool))]
            words = base.split()
            rng.shuffle(words)
            texts.append(" ".join(words))
            labels.append(lab)
    return texts, np.array(labels)


def test_toy_model_learns_and_is_deterministic(toy_corpus):
    """T-MODEL-01: пайплайн обучается, качество выше мажоритарного, детерминизм при одном seed."""
    texts, y = toy_corpus
    f, d, t = group_split(texts, y, seed=42)
    m1 = build_pipeline("word_only")
    m1.fit([texts[i] for i in f], y[f])
    p1 = m1.predict_proba([texts[i] for i in t])
    m2 = build_pipeline("word_only")
    m2.fit([texts[i] for i in f], y[f])
    p2 = m2.predict_proba([texts[i] for i in t])
    assert np.allclose(p1, p2)
    assert np.allclose(p1.sum(axis=1), 1.0, atol=1e-6)
    metrics = evaluate(y[t], p1.argmax(axis=1))
    majority = np.bincount(y[f]).max() / len(f)
    assert metrics["accuracy"] > majority


def test_blend_and_weight_selection(toy_corpus):
    """T-MODEL-02: смесь вероятностей нормирована; подбор весов даёт сумму 1."""
    texts, y = toy_corpus
    f, d, t = group_split(texts, y, seed=42)
    probas = []
    for cfg in ("word_only", "char_only", "union_weighted"):
        m = build_pipeline(cfg)
        m.fit([texts[i] for i in f], y[f])
        probas.append(m.predict_proba([texts[i] for i in d]))
    w, f1 = select_weights(probas, y[d])
    assert abs(sum(w) - 1.0) < 1e-6
    assert 0 <= f1 <= 1
    blended = blend_proba(probas, w)
    assert np.allclose(blended.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------- политика


def test_decide_policy_thresholds():
    """T-POL-01: ниже порога — uncertain (не normal); спам требует повышенного порога."""
    proba = np.array(
        [
            [0.10, 0.80, 0.10],  # уверенный спам
            [0.10, 0.60, 0.30],  # спам 0.60 < tau_spam 0.65 -> uncertain
            [0.50, 0.30, 0.20],  # normal 0.50 < tau 0.55 -> uncertain
            [0.05, 0.05, 0.90],  # уверенный кликбейт
        ]
    )
    d = decide(proba)
    assert d[0]["status"] == "classified" and d[0]["label"] == "spam"
    assert d[1]["status"] == "uncertain" and d[1]["label"] is None
    assert d[2]["status"] == "uncertain" and d[2]["reason"] == "low_confidence"
    assert d[3]["label"] == "clickbait"


def test_policy_metrics_consistency(toy_corpus):
    """T-POL-02: coverage и счётчики согласованы; accuracy_all не зависит от порога."""
    texts, y = toy_corpus
    f, d, t = group_split(texts, y, seed=42)
    m = build_pipeline("word_only")
    m.fit([texts[i] for i in f], y[f])
    p = m.predict_proba([texts[i] for i in t])
    pm = policy_metrics(y[t], p)
    assert pm["n_classified"] + pm["n_uncertain"] == len(t)
    assert 0 <= pm["coverage"] <= 1
    if pm["n_classified"]:
        assert pm["accuracy_classified"] >= pm["accuracy_all"] - 1e-9


# ---------------------------------------------------------------- дрейф


def test_psi_stable_vs_shifted():
    """T-DRIFT-01: PSI ~0 на одинаковых распределениях и >0.25 при сдвиге."""
    rng = np.random.default_rng(1)
    ref = rng.normal(20, 5, 5000)
    same = rng.normal(20, 5, 5000)
    shifted = rng.normal(60, 5, 5000)
    assert psi(ref, same) < 0.1
    assert psi(ref, shifted) > 0.25
    assert psi_level(0.05) == "stable"
    assert psi_level(0.15) == "moderate"
    assert psi_level(0.5) == "significant"


def test_ks_detects_shift():
    """T-DRIFT-02: KS-тест различает сдвинутые распределения (p < 0.05)."""
    rng = np.random.default_rng(2)
    ref = rng.normal(20, 5, 2000)
    shifted = rng.normal(30, 5, 2000)
    res = ks_drift(ref, shifted)
    assert res["p_value"] < 0.05
    assert res["ks_stat"] > 0.2