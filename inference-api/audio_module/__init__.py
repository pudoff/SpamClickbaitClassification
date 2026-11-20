"""Аудио-модуль: ASR (инференс) и TTS-аугментация (офлайн-обучение)."""

from .audio_io import AudioDecodeError, AudioError, AudioTooLongError
from .schemas import ASRConfig, ASRResult, Segment

__all__ = ["ASRConfig", "ASRResult", "Segment", "AudioError", "AudioDecodeError", "AudioTooLongError"]
