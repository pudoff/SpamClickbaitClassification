"""Ограничения загрузки и преобразование результата ASR для HTTP."""

import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import HTTPException, Request, UploadFile
from pydantic import ValidationError

from app.schemas import PredictRequest, TranscriptionResponse
from audio_module.audio_io import AudioDecodeError, AudioTooLongError, UnsupportedAudioError
from audio_module.stt_service import STTService

MAX_AUDIO_BYTES = 20 * 1024 * 1024
logger = logging.getLogger("inference-api")


def load_stt(num_threads: int) -> STTService:
    import torch

    torch.set_num_threads(num_threads)
    return STTService()


def transcribe(audio: UploadFile, request: Request) -> TranscriptionResponse:
    if audio.size is not None and audio.size > MAX_AUDIO_BYTES:
        raise HTTPException(413, detail="payload_too_large")
    stt = request.app.state.stt
    if stt is None:
        raise HTTPException(503, detail="asr_unavailable")

    lock = request.app.state.stt_lock
    if not lock.acquire(timeout=30):
        raise HTTPException(503, detail="overloaded", headers={"Retry-After": "5"})
    try:
        with TemporaryDirectory(prefix="inference-audio-") as directory:
            path = Path(directory) / "upload"
            uploaded = 0
            with path.open("wb") as out:
                while chunk := audio.file.read(1024 * 1024):
                    uploaded += len(chunk)
                    if uploaded > MAX_AUDIO_BYTES:
                        raise HTTPException(413, detail="payload_too_large")
                    out.write(chunk)
            result = stt.transcribe(path)
    except UnsupportedAudioError as exc:
        raise HTTPException(415, detail="unsupported_audio_format") from exc
    except AudioDecodeError as exc:
        raise HTTPException(422, detail="audio_decode_failed") from exc
    except AudioTooLongError as exc:
        raise HTTPException(422, detail="audio_too_long") from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("сбой распознавания: %s", type(exc).__name__)
        raise HTTPException(503, detail="asr_unavailable") from exc
    finally:
        lock.release()

    transcript = None
    reason = result.reason
    if result.status == "ok":
        try:
            transcript = PredictRequest(text=result.text[:20_000]).text
        except ValidationError:
            reason = "no_speech"

    return TranscriptionResponse(
        status="transcribed" if transcript else "not_classified",
        reason=reason,
        detected_language=result.language,
        audio_duration_s=result.duration_s,
        speech_duration_s=result.speech_s,
        transcript=transcript,
        asr_model=f"{result.asr_model}@1",
        lid_model=f"{result.lid_model}@1",
        timings_ms={key: result.timings_ms.get(key, 0) for key in ("decode", "vad", "lid", "asr", "total")},
    )
