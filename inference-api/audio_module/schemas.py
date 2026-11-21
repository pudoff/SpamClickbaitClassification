from __future__ import annotations

from dataclasses import asdict, dataclass, field

STATUS_OK = "ok"
STATUS_NOT_CLASSIFIED = "not_classified"

REASON_NO_SPEECH = "no_speech"
REASON_SPEECH_TOO_SHORT = "speech_too_short"
REASON_UNSUPPORTED_LANGUAGE = "unsupported_language"


@dataclass
class ASRConfig:
    sample_rate: int = 16_000
    max_duration_s: float = 300.0  # лимит бота - duration <= 300
    ffmpeg_timeout_s: float = 15.0
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 300
    max_chunk_s: float = 20.0  # лимит transcribe у GigaAM - 25
    min_speech_s: float = 1.0
    lid_max_s: float = 30.0
    lid_reject_prob: float = 0.70


@dataclass
class Segment:
    start_s: float
    end_s: float
    text: str = ""


@dataclass
class ASRResult:
    status: str
    reason: str | None
    text: str = ""
    language: str | None = None
    language_prob_ru: float | None = None
    duration_s: float = 0.0
    speech_s: float = 0.0
    asr_model: str | None = None
    lid_model: str | None = None
    segments: list[Segment] = field(default_factory=list)
    timings_ms: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
