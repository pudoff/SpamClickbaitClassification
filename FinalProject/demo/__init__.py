"""Демонстрационный пайплайн итоговой защиты SpamClickbaitClassification.

Модули:
- data: загрузка df.csv, нормализация, валидация, групповой сплит без утечек;
- models: конфигурации TF-IDF + LogisticRegression, политика отказа, метрики;
- drift: PSI и KS-тест для мониторинга дрейфа (по образцу практики В13 курса).

Код самодостаточен (не импортирует inference-api), чтобы ноутбук работал в Colab.
Нормализация demo-clean-v1 — упрощённый вариант clean-text-v2 из app/text.py.
"""

from demo.data import (  # noqa: F401
    clean_text,
    group_split,
    load_dataset,
    template_key,
    validate_dataset,
)
from demo.drift import ks_drift, psi  # noqa: F401
from demo.models import (  # noqa: F401
    CONFIGS,
    LABEL_NAMES,
    build_pipeline,
    decide,
    evaluate,
    policy_metrics,
)