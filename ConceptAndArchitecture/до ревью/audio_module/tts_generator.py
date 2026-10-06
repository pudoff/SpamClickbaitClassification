from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from .audio_io import encode_opus, read_wav, write_wav

SILERO_SPEAKERS = ["aidar", "baya", "kseniya", "xenia", "eugene"]
MAX_CHARS = 800


def sanitize_text(text: str, min_words: int = 3) -> str:
    def num(m: re.Match) -> str:
        try:
            from num2words import num2words
            return " " + num2words(int(m.group()), lang="ru") + " "
        except Exception:
            return " "
    t = re.sub(r"\d{1,9}", num, str(text))
    t = re.sub(r"[^А-Яа-яЁё\s.,!?:\-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > MAX_CHARS:
        t = t[:MAX_CHARS].rsplit(" ", 1)[0]
    return t if len(t.split()) >= min_words else ""


class TTSBackend(Protocol):
    name: str
    sample_rate: int
    speakers: list[str]

    def synth(self, text: str, speaker: str) -> np.ndarray: ...


class SileroBackend:
    name = "silero_v4_ru"
    sample_rate = 24_000
    speakers = SILERO_SPEAKERS

    def __init__(self, device: Optional[str] = None):
        import torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, _ = torch.hub.load("snakers4/silero-models", "silero_tts",
                                       language="ru", speaker="v4_ru", trust_repo=True)
        self.model.to(self.device)

    def synth(self, text: str, speaker: str) -> np.ndarray:
        audio = self.model.apply_tts(text=text, speaker=speaker, sample_rate=self.sample_rate,
                                     put_accent=True, put_yo=True)
        return audio.detach().cpu().numpy().astype(np.float32)


class PiperBackend:
    name = "piper"

    def __init__(self, voices: dict[str, str], piper_bin: str = "piper"):
        self.voices, self.piper_bin = voices, piper_bin
        self.speakers = list(voices)
        self.sample_rate = 22_050

    def synth(self, text: str, speaker: str) -> np.ndarray:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "o.wav"
            subprocess.run([self.piper_bin, "--model", self.voices[speaker], "--output_file", str(out)],
                           input=text.encode(), check=True, capture_output=True, timeout=120)
            wave, self.sample_rate = read_wav(out)
        return wave


@dataclass
class AugParams:
    speed: float
    snr_db: Optional[float]
    opus_kbps: int


def sample_params(rng: np.random.Generator) -> AugParams:
    return AugParams(
        speed=float(round(rng.uniform(0.9, 1.1), 3)),
        snr_db=None if rng.random() < 0.2 else float(round(rng.uniform(10, 30), 1)),
        opus_kbps=int(rng.choice([16, 24, 32])),
    )


def add_noise(wave: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    power = float(np.mean(wave ** 2)) + 1e-12
    noise = rng.normal(0.0, np.sqrt(power / 10 ** (snr_db / 10)), size=wave.shape)
    return np.clip(wave + noise.astype(np.float32), -1.0, 1.0)


class TTSAugmenter:
    def __init__(self, backend: TTSBackend | str = "silero", speaker: str = "aidar"):
        self.backend = SileroBackend() if backend == "silero" else backend
        self.speaker = speaker

    def text_to_speech(self, text: str, output_audio_path: str = "generated_spam.wav",
                       speaker: Optional[str] = None) -> str:
        clean = sanitize_text(text)
        if not clean:
            raise ValueError("после очистки текст пуст или короче 3 слов")
        wave = self.backend.synth(clean, speaker or self.speaker)
        return write_wav(output_audio_path, wave, self.backend.sample_rate)

    def synthesize_augmented(self, text: str, out_path: str | Path, speaker: Optional[str] = None,
                             rng: Optional[np.random.Generator] = None) -> dict:
        rng = rng or np.random.default_rng()
        clean = sanitize_text(text)
        if not clean:
            raise ValueError("текст пуст после очистки")
        speaker = speaker or self.speaker
        wave = self.backend.synth(clean, speaker)
        p = sample_params(rng)
        if p.snr_db is not None:
            wave = add_noise(wave, p.snr_db, rng)
        encode_opus(wave, self.backend.sample_rate, out_path, kbps=p.opus_kbps, speed=p.speed)
        return {"tts_model": self.backend.name, "speaker": speaker, "tts_text": clean, **asdict(p)}


if __name__ == "__main__":
    tts = TTSAugmenter()
    print(tts.text_to_speech("Вы выиграли приз, перейдите по ссылке и получите деньги.", "sample_spam.wav"))
