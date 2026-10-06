from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000


class AudioError(Exception):
    code = "audio_error"


class AudioDecodeError(AudioError):
    code = "decode_failed"


class AudioTooLongError(AudioError):
    code = "too_long"


def _run(cmd: list[str], timeout: float, stdin: bytes | None = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, input=stdin, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as e:
        raise AudioDecodeError(f"{cmd[0]}: timeout {timeout}s") from e
    except FileNotFoundError as e:
        raise AudioError(f"{cmd[0]} не установлен (apt install ffmpeg)") from e


def probe_duration(path: str | Path, timeout: float = 15.0) -> float:
    p = _run(
        ["ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe",
         "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        timeout,
    )
    if p.returncode != 0:
        raise AudioDecodeError(p.stderr.decode(errors="ignore").strip() or "ffprobe failed")
    try:
        return float(p.stdout.decode().strip())
    except ValueError as e:
        raise AudioDecodeError("не удалось определить длительность") from e


def decode_audio(path: str | Path, sample_rate: int = SAMPLE_RATE, timeout: float = 15.0) -> np.ndarray:
    p = _run(
        ["ffmpeg", "-v", "error", "-protocol_whitelist", "file,pipe", "-i", str(path),
         "-vn", "-f", "s16le", "-ac", "1", "-ar", str(sample_rate), "-"],
        timeout,
    )
    if p.returncode != 0 or not p.stdout:
        raise AudioDecodeError(p.stderr.decode(errors="ignore").strip() or "пустой результат декодирования")
    return np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def write_wav(path: str | Path, wave_f32: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    pcm = (np.clip(wave_f32, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return str(path)


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return (data.reshape(-1, ch).mean(axis=1) if ch > 1 else data), sr


def encode_opus(wave_f32: np.ndarray, sample_rate: int, out_path: str | Path,
                kbps: int = 24, speed: float = 1.0, timeout: float = 60.0) -> str:
    pcm = (np.clip(wave_f32, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0"]
    if abs(speed - 1.0) > 1e-3:
        cmd += ["-filter:a", f"atempo={speed:.3f}"]
    cmd += ["-c:a", "libopus", "-b:a", f"{kbps}k", "-ar", "16000", "-ac", "1", str(out_path)]
    p = _run(cmd, timeout, stdin=pcm)
    if p.returncode != 0:
        raise AudioError(p.stderr.decode(errors="ignore").strip() or "ffmpeg encode failed")
    return str(out_path)
