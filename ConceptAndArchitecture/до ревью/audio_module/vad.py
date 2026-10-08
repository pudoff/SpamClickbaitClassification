from __future__ import annotations

import numpy as np

from .schemas import ASRConfig


def merge_segments(ts: list[tuple[float, float]], max_len: float = 20.0) -> list[tuple[float, float]]:
    pieces: list[tuple[float, float]] = []
    for s, e in ts:
        while e - s > max_len:
            pieces.append((s, s + max_len))
            s += max_len
        if e > s:
            pieces.append((s, e))
    out: list[tuple[float, float]] = []
    for s, e in pieces:
        if out and e - out[-1][0] <= max_len:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


class SileroVAD:
    def __init__(self, cfg: ASRConfig | None = None):
        from silero_vad import load_silero_vad  # lazy: тяжёлый импорт
        self.cfg = cfg or ASRConfig()
        self.model = load_silero_vad()

    def detect(self, wave: np.ndarray) -> list[tuple[float, float]]:
        import torch
        from silero_vad import get_speech_timestamps
        c = self.cfg
        ts = get_speech_timestamps(
            torch.from_numpy(wave), self.model, sampling_rate=c.sample_rate,
            threshold=c.vad_threshold, min_speech_duration_ms=c.vad_min_speech_ms,
            min_silence_duration_ms=c.vad_min_silence_ms,
            max_speech_duration_s=c.max_chunk_s, return_seconds=True,
        )
        return [(float(t["start"]), float(t["end"])) for t in ts]
