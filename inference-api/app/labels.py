"""Классы модели: кодировка датасета Datasets/df.csv, поле is_spam1_or_clickbait2."""

LABELS: dict[int, str] = {0: "normal", 1: "spam", 2: "clickbait"}
# Порядок столбцов predict_proba и поле classes манифеста ансамбля
CLASS_IDS: tuple[int, ...] = (0, 1, 2)
LABEL_IDS: dict[str, int] = {name: idx for idx, name in LABELS.items()}
N_CLASSES = len(CLASS_IDS)
