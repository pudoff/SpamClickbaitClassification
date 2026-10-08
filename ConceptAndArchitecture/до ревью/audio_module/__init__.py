"""Аудио-модуль: ASR (инференс) и TTS-аугментация (офлайн-обучение)."""
from .schemas import ASRConfig, ASRResult, Segment
from .audio_io import AudioError, AudioDecodeError, AudioTooLongError

__all__ = ["ASRConfig", "ASRResult", "Segment", "AudioError", "AudioDecodeError", "AudioTooLongError"]
