import numpy as np
import pytest

from audio_module.audio_io import (AudioDecodeError, AudioTooLongError, decode_audio,
                                   encode_opus, probe_duration, write_wav)
from audio_module.benchmark_asr import wer
from audio_module.schemas import ASRConfig
from audio_module.stt_service import STTService
from audio_module.tts_generator import add_noise, sample_params, sanitize_text
from audio_module.vad import merge_segments

SR = 16_000


def tone(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


@pytest.fixture()
def ogg(tmp_path):
    p = tmp_path / "v.ogg"
    encode_opus(tone(6.0), SR, p, kbps=24)
    return p


class FakeVAD:
    def __init__(self, ts): self.ts = ts
    def detect(self, wave): return self.ts


class FakeLID:
    name = "fake-lid"
    def __init__(self, p_ru): self.p = p_ru
    def detect(self, wave): return ("ru" if self.p > .5 else "en"), self.p


class FakeASR:
    name = "fake-asr"
    def __init__(self): self.calls = 0
    def transcribe_file(self, path):
        self.calls += 1
        return f"кусок {self.calls}"


def svc(ts, p_ru=0.99, **kw):
    return STTService(FakeASR(), vad=FakeVAD(ts), lid=FakeLID(p_ru), **kw)


def test_opus_roundtrip_and_probe(ogg):
    assert abs(probe_duration(ogg) - 6.0) < 0.2
    assert abs(len(decode_audio(ogg)) / SR - 6.0) < 0.2


def test_decode_error(tmp_path):
    bad = tmp_path / "bad.ogg"
    bad.write_bytes(b"not audio")
    with pytest.raises(AudioDecodeError):
        probe_duration(bad)


def test_too_long(ogg):
    with pytest.raises(AudioTooLongError):
        svc([(0, 5)], cfg=ASRConfig(max_duration_s=2.0)).transcribe(ogg)


def test_no_speech(ogg):
    r = svc([]).transcribe(ogg)
    assert (r.status, r.reason) == ("not_classified", "no_speech")


def test_speech_too_short(ogg):
    r = svc([(0.0, 0.6)]).transcribe(ogg)
    assert (r.status, r.reason) == ("not_classified", "speech_too_short")


def test_unsupported_language(ogg):
    r = svc([(0.0, 5.0)], p_ru=0.1).transcribe(ogg)
    assert (r.status, r.reason) == ("not_classified", "unsupported_language")


def test_ok_and_contract(ogg):
    r = svc([(0.0, 2.0), (2.5, 5.0)]).transcribe(ogg)
    assert r.status == "ok" and r.reason is None
    assert r.text == "кусок 1" and r.asr_model == "fake-asr"   # оба куска влезли в один сегмент <=20 с
    d = r.to_dict()
    for k in ("status", "reason", "text", "language", "duration_s", "asr_model", "lid_model", "timings_ms", "segments"):
        assert k in d


def test_merge_segments():
    assert merge_segments([(0, 5), (6, 12), (13, 19), (21, 30)], 20) == [(0, 19), (21, 30)]
    assert merge_segments([(0, 45)], 20) == [(0, 20), (20, 40), (40, 45)]


def test_wer():
    assert wer("Привет, мир!", "привет мир") == 0.0
    assert wer("один два три четыре", "один два три") == 0.25
    assert wer("ёлка", "елка") == 0.0


def test_sanitize_text():
    assert sanitize_text("Купи iPhone за 5 рублей сейчас!") != ""
    assert "iPhone" not in sanitize_text("Купи iPhone за 5 рублей сейчас!")
    assert sanitize_text("ok") == ""


def test_augmentation_and_encode(tmp_path):
    rng = np.random.default_rng(1)
    p = sample_params(rng)
    assert 0.9 <= p.speed <= 1.1 and p.opus_kbps in (16, 24, 32)
    noisy = add_noise(tone(1.0), 20.0, rng)
    assert noisy.shape == tone(1.0).shape and np.abs(noisy).max() <= 1.0
    out = tmp_path / "a.ogg"
    encode_opus(tone(3.0), SR, out, kbps=16, speed=1.1)
    assert 2.5 < probe_duration(out) < 3.0
