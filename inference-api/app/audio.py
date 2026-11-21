from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal

from fastapi import HTTPException, Request, Response, UploadFile
from fastapi.routing import APIRoute
from pydantic import ValidationError
from starlette.types import Message

from app.schemas import PredictRequest
from audio_module.audio_io import AudioDecodeError, AudioTooLongError, UnsupportedAudioError

if TYPE_CHECKING:
    from audio_module.stt_service import STTService

MAX_AUDIO_BYTES = 20 * 1024 * 1024
MAX_MULTIPART_OVERHEAD = 64 * 1024
logger = logging.getLogger("inference-api")


@dataclass(frozen=True)
class Transcription:
    reason: Literal["no_speech", "speech_too_short", "unsupported_language"] | None
    detected_language: str | None
    audio_duration_s: float
    speech_duration_s: float
    transcript: str | None
    asr_model: str
    lid_model: str
    timings_ms: dict[str, int]


class AudioUploadRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Awaitable[Response]]:
        handle = super().get_route_handler()

        async def bounded_upload(request: Request) -> Response:
            max_body_bytes = MAX_AUDIO_BYTES + MAX_MULTIPART_OVERHEAD
            try:
                declared_size = int(request.headers.get("content-length", "0"))
            except ValueError as exc:
                raise HTTPException(400, detail="invalid_content_length") from exc
            if declared_size > max_body_bytes:
                raise HTTPException(413, detail="payload_too_large")

            received = 0

            async def receive() -> Message:
                nonlocal received
                message = await request.receive()
                if message["type"] == "http.request":
                    received += len(message.get("body", b""))
                    if received > max_body_bytes:
                        raise HTTPException(413, detail="payload_too_large")
                return message

            return await handle(Request(request.scope, receive))

        return bounded_upload


def load_stt(num_threads: int) -> STTService:
    import torch

    from audio_module.stt_service import STTService

    torch.set_num_threads(num_threads)
    return STTService()


def transcribe(audio: UploadFile, request: Request) -> Transcription:
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

    return Transcription(
        reason=reason,
        detected_language=result.language,
        audio_duration_s=result.duration_s,
        speech_duration_s=result.speech_s,
        transcript=transcript,
        asr_model=f"{result.asr_model}@1",
        lid_model=f"{result.lid_model}@1",
        timings_ms={key: result.timings_ms.get(key, 0) for key in ("decode", "vad", "lid", "asr", "total")},
    )
