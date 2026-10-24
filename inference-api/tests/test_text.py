"""Нормализация: v1 неизменна (на ней обучен ансамбль), v2 исправляет ссылки КАПСОМ."""

import pytest

from app.text import (
    NORMALIZER_V1,
    NORMALIZER_V2,
    clean_text,
    clean_text_v2,
    get_normalizer,
    normalizer_of,
)
from tests.conftest import TOY_DATA

# Эталон v1, снятый до введения v2: любое изменение clean_text ломает выпущенный ансамбль.
V1_GOLDEN = [
    ("Вам одобрен кредит 300 000 руб! Звоните 8-800", "вам одобрен кредит NUM NUM руб звоните NUM NUM"),
    ("СМОТРИ HTTP://SPAM.RU/PROMO и WWW.SITE.COM", "смотри http spam ru promo и www site com"),
    ("смотри http://spam.ru/promo и www.site.com", "смотри url и url"),
    ("<b>Ёлка</b> @user #тег 3,14", "елка NUM"),
    ("Http://Mixed.Case/x", "http mixed case x"),
    ("", ""),
]


@pytest.mark.parametrize(("raw", "expected"), V1_GOLDEN)
def test_v1_unchanged(raw, expected):
    assert clean_text(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("СМОТРИ HTTP://SPAM.RU/PROMO и WWW.SITE.COM", "смотри url и url"),
        ("Http://Mixed.Case/x", "url"),
        ("ПЕРЕХОДИ <B>СЕЙЧАС</B> @USER", "переходи сейчас"),
    ],
)
def test_v2_masks_links_in_any_case(raw, expected):
    assert clean_text_v2(raw) == expected
    assert clean_text_v2(raw) == clean_text_v2(raw.lower())


def test_v2_equals_v1_without_uppercase_links():
    texts = [text for text, _ in TOY_DATA] + [raw for raw, _ in V1_GOLDEN if "HTTP" not in raw and "Http" not in raw]
    assert [clean_text_v2(t) for t in texts] == [clean_text(t) for t in texts]


def test_normalizer_lookup():
    assert get_normalizer(NORMALIZER_V1) is clean_text and get_normalizer(NORMALIZER_V2) is clean_text_v2
    with pytest.raises(ValueError, match="нормализация"):
        get_normalizer("v3")
    assert normalizer_of({}) == normalizer_of({"config": None}) == NORMALIZER_V1
    assert normalizer_of({"config": {"normalizer": NORMALIZER_V2}}) == NORMALIZER_V2
