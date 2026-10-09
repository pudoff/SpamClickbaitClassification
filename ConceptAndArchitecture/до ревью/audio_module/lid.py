from __future__ import annotations

import numpy as np


class LanguageDetector:
    name = "faster-whisper-base"

    def __init__(self, model_size: str = "base"):
        from faster_whisper import WhisperModel  # lazy
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def detect(self, wave: np.ndarray) -> tuple[str, float]:
        lang, prob, all_probs = self.model.detect_language(audio=wave)
        p_ru = dict(all_probs).get("ru", prob if lang == "ru" else 0.0)
        return lang, float(p_ru)
