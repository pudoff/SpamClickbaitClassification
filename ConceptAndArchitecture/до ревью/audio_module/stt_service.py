from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from . import schemas as S
from .audio_io import AudioTooLongError, decode_audio, probe_duration, write_wav
from .vad import merge_segments


class ASRBackend(Protocol):
    name: str

    def transcribe_file(self, wav_path: str) -> str: ...


class GigaAMBackend:

    def __init__(self, model_name: str = "v3_e2e_rnnt", device: Optional[str] = None):
        import gigaam
        import torch
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = gigaam.load_model(model_name, fp16_encoder=(device == "cuda"), device=device)
        self.name = f"gigaam-{model_name}"

    def transcribe_file(self, wav_path: str) -> str:
        r = self.model.transcribe(wav_path)  # лимит 25 с — сегменты режет VAD (<=20 с)
        return str(getattr(r, "text", r)).strip()


class WhisperBackend:
    def __init__(self, model_size: str = "large-v3-turbo", device: Optional[str] = None):
        from faster_whisper import WhisperModel
        import torch
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = WhisperModel(model_size, device=device,
                                  compute_type="float16" if device == "cuda" else "int8")
        self.name = f"faster-whisper-{model_size}"

    def transcribe_file(self, wav_path: str) -> str:
        segs, _ = self.model.transcribe(wav_path, language="ru", beam_size=5,
                                        vad_filter=False, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segs).strip()


def build_backend(name: str = "gigaam", **kw) -> ASRBackend:
    name = name.lower()
    if name == "gigaam":
        return GigaAMBackend(**kw)
    if name in ("whisper", "faster-whisper"):
        return WhisperBackend(**kw)
    raise ValueError(f"неизвестный backend: {name} (gigaam | whisper)")


class STTService:
    def __init__(self, backend: ASRBackend | str = "gigaam", vad=None, lid=None,
                 cfg: S.ASRConfig | None = None, use_lid: bool = True):
        self.cfg = cfg or S.ASRConfig()
        self.backend = build_backend(backend) if isinstance(backend, str) else backend
        if vad is None:
            from .vad import SileroVAD
            vad = SileroVAD(self.cfg)
        self.vad = vad
        if lid is None and use_lid:
            from .lid import LanguageDetector
            lid = LanguageDetector()
        self.lid = lid

    def transcribe(self, audio_path: str | Path) -> S.ASRResult:
        c, t = self.cfg, {}
        t0 = time.perf_counter()

        def tick(key: str, since: float) -> float:
            now = time.perf_counter()
            t[key] = int((now - since) * 1000)
            return now

        duration = probe_duration(audio_path, c.ffmpeg_timeout_s)
        if duration > c.max_duration_s:
            raise AudioTooLongError(f"{duration:.0f} с > {c.max_duration_s:.0f} с")
        wave = decode_audio(audio_path, c.sample_rate, c.ffmpeg_timeout_s)
        now = tick("decode", t0)

        raw = self.vad.detect(wave)
        speech_s = float(sum(e - s for s, e in raw))
        now = tick("vad", now)
        base = dict(duration_s=round(duration, 2), speech_s=round(speech_s, 2),
                    asr_model=self.backend.name,
                    lid_model=getattr(self.lid, "name", None))

        if not raw:
            return S.ASRResult(S.STATUS_NOT_CLASSIFIED, S.REASON_NO_SPEECH, timings_ms=t, **base)
        if speech_s < c.min_speech_s:
            return S.ASRResult(S.STATUS_NOT_CLASSIFIED, S.REASON_SPEECH_TOO_SHORT, timings_ms=t, **base)

        language, p_ru = "ru", None
        if self.lid is not None:
            language, p_ru = self.lid.detect(self._lid_audio(wave, raw))
            now = tick("lid", now)
            if (1.0 - p_ru) >= c.lid_reject_prob:
                return S.ASRResult(S.STATUS_NOT_CLASSIFIED, S.REASON_UNSUPPORTED_LANGUAGE,
                                   language=language, language_prob_ru=round(p_ru, 3),
                                   timings_ms=t, **base)

        segments: list[S.Segment] = []
        with tempfile.TemporaryDirectory(prefix="asr_") as tmp:
            for i, (s, e) in enumerate(merge_segments(raw, c.max_chunk_s)):
                chunk = wave[int(s * c.sample_rate): int(e * c.sample_rate)]
                path = write_wav(Path(tmp) / f"seg{i}.wav", chunk, c.sample_rate)
                segments.append(S.Segment(round(s, 2), round(e, 2), self.backend.transcribe_file(path)))
        tick("asr", now)
        t["total"] = int((time.perf_counter() - t0) * 1000)

        text = " ".join(s.text for s in segments if s.text).strip()
        return S.ASRResult(S.STATUS_OK, None, text=text, language=language,
                           language_prob_ru=None if p_ru is None else round(p_ru, 3),
                           segments=segments, timings_ms=t, **base)

    def _lid_audio(self, wave: np.ndarray, ts: list[tuple[float, float]]) -> np.ndarray:
        sr, budget, parts = self.cfg.sample_rate, self.cfg.lid_max_s, []
        for s, e in ts:
            take = min(e - s, budget)
            parts.append(wave[int(s * sr): int((s + take) * sr)])
            budget -= take
            if budget <= 0:
                break
        return np.concatenate(parts)


if __name__ == "__main__":
    import json
    import sys
    if len(sys.argv) < 2:
        sys.exit("usage: python -m audio_module.stt_service <audio> [gigaam|whisper]")
    svc = STTService(sys.argv[2] if len(sys.argv) > 2 else "gigaam")
    print(json.dumps(svc.transcribe(sys.argv[1]).to_dict(), ensure_ascii=False, indent=2))
