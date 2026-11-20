from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import joblib
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from audio_module.audio_io import AudioDecodeError, AudioTooLongError, UnsupportedAudioError
from audio_module.schemas import ASRResult

TRANSCRIPT = "Буду через десять минут, возьми хлеба"
UPLOAD = {"audio": ("проверка.ogg", b"test audio")}


@pytest.fixture
def audio_client(tmp_path, make_settings, toy_lr, monkeypatch):
    artifact = tmp_path / "model.joblib"
    joblib.dump(toy_lr, artifact)
    asr = Mock()
    monkeypatch.setattr("app.main.load_stt", lambda _: asr)
    with TestClient(create_app(replace(make_settings(artifact), asr_enabled=True))) as client:
        yield client, asr


def test_audio_classification_reuses_text_policy(audio_client):
    client, asr = audio_client
    paths = []

    def transcribe(path: Path) -> ASRResult:
        paths.append(path)
        assert path.read_bytes() == b"test audio"
        return ASRResult("ok", None, TRANSCRIPT, "ru", duration_s=3.1, speech_s=2.5)

    asr.transcribe.side_effect = transcribe
    response = client.post("/v1/classifications", files=UPLOAD, headers={"X-Request-Id": "test-audio"})
    assert response.status_code == 200
    speech = response.json()
    text = client.post("/predict", json={"text": TRANSCRIPT}).json()
    assert response.headers["X-Request-Id"] == speech["request_id"] == "test-audio"
    assert speech["transcript"] == TRANSCRIPT
    assert speech["models"]["classifier_model"] == text["model_version"]
    for field in ("status", "label", "reason", "confidence", "scores"):
        assert speech[field] == text[field]
    raw = client.post("/v1/transcriptions", files=UPLOAD)
    assert raw.status_code == 200
    assert raw.json()["status"] == "transcribed"
    assert raw.json()["transcript"] == TRANSCRIPT
    assert len(paths) == 2 and all(not path.exists() for path in paths)


@pytest.mark.parametrize("reason", ["no_speech", "speech_too_short", "unsupported_language"])
def test_audio_without_recognizable_speech_has_no_label(audio_client, reason):
    client, asr = audio_client
    asr.transcribe.return_value = ASRResult("not_classified", reason)
    response = client.post("/v1/classifications", files=UPLOAD)
    assert response.status_code == 200
    body = response.json()
    assert (body["status"], body["reason"]) == ("not_classified", reason)
    assert all(body[key] is None for key in ("transcript", "label", "scores", "confidence"))


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (AudioDecodeError, 422, "audio_decode_failed"),
        (AudioTooLongError, 422, "audio_too_long"),
        (UnsupportedAudioError, 415, "unsupported_audio_format"),
        (RuntimeError, 503, "asr_unavailable"),
    ],
)
def test_audio_error_removes_temporary_file(audio_client, error, status, code):
    client, asr = audio_client
    asr.transcribe.side_effect = error
    response = client.post("/v1/transcriptions", files=UPLOAD)
    assert response.status_code == status
    assert response.json() == {"detail": code}
    assert not asr.transcribe.call_args.args[0].exists()


def test_audio_upload_size_checked_before_asr(audio_client, monkeypatch):
    client, asr = audio_client
    monkeypatch.setattr("app.audio.MAX_AUDIO_BYTES", 5)
    response = client.post("/v1/transcriptions", files=UPLOAD)
    assert response.status_code == 413
    asr.transcribe.assert_not_called()
