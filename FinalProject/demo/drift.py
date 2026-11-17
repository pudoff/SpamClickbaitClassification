"""Дрейф данных: PSI и KS-тест (по образцу практики В13 курса).

PSI (Population Stability Index) по бинам эталонного распределения:
< 0.1 — стабильно; 0.1–0.25 — умеренный сдвиг; > 0.25 — значимый дрейф.
Дрейф — сигнал к дополнительной разметке, а не к автоматическому переобучению.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # вырожденное распределение (почти константа)
        edges = np.array([reference.min() - 1, reference.min(), reference.max() + 1])
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(reference, bins=edges)[0] / max(len(reference), 1)
    cur_counts = np.histogram(current, bins=edges)[0] / max(len(current), 1)
    ref_counts = np.clip(ref_counts, eps, None)
    cur_counts = np.clip(cur_counts, eps, None)
    return float(np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts)))


def psi_level(value: float) -> str:
    if value < 0.1:
        return "stable"
    if value < 0.25:
        return "moderate"
    return "significant"


def ks_drift(reference: np.ndarray, current: np.ndarray) -> dict:
    res = stats.ks_2samp(np.asarray(reference), np.asarray(current))
    return {"ks_stat": round(float(res.statistic), 4), "p_value": float(res.pvalue)}