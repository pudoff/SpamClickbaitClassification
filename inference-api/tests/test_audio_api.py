import json
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
def audio_settings(tmp_path, make_settings, toy_lr):
    artifact = tmp_path / "model.joblib"
    joblib.dump(toy_lr, artifact)
    return replace(make_settings(artifact), asr_enabled=True)


@pytest.fixture
def audio_client(audio_settings, monkeypatch):
    asr = Mock()
    monkeypatch.setattr("app.main.load_stt", lambda _: asr)
    with TestClient(create_app(audio_settings)) as client:
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


def test_audio_uses_calibrated_policy_from_model_metadata(audio_settings, monkeypatch):
    config = {"voice_policy": {"tau": 0.0, "tau_spam": 0.0}}
    audio_settings.meta_path.write_text(json.dumps({"config": config}), encoding="utf-8")
    asr = Mock()
    asr.transcribe.return_value = ASRResult("ok", None, TRANSCRIPT, "ru")
    monkeypatch.setattr("app.main.load_stt", lambda _: asr)
    with TestClient(create_app(replace(audio_settings, tau=1.0, tau_spam=1.0))) as client:
        assert client.get("/model-info").json()["config"]["voice_policy"] == config["voice_policy"]
        assert client.post("/predict", json={"text": TRANSCRIPT}).json()["status"] == "uncertain"
        assert client.post("/v1/classifications", files=UPLOAD).json()["status"] == "classified"


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


@pytest.mark.parametrize("endpoint", ["/v1/transcriptions", "/v1/classifications"])
def test_oversized_content_length_is_rejected_before_body_is_read(audio_client, monkeypatch, endpoint):
    client, asr = audio_client
    monkeypatch.setattr("app.audio.MAX_AUDIO_BYTES", 10)
    monkeypatch.setattr("app.audio.MAX_MULTIPART_OVERHEAD", 0)

    def body():
        pytest.fail("Сервис попытался прочитать запрос, который превышает лимит")
        yield b""

    result = client.post(endpoint, headers={"Content-Length": "11"}, content=body())
    assert result.status_code == 413
    assert result.json() == {"detail": "payload_too_large"}
    asr.transcribe.assert_not_called()


def test_multipart_body_without_content_length_is_also_limited(audio_client, monkeypatch):
    client, asr = audio_client
    monkeypatch.setattr("app.audio.MAX_AUDIO_BYTES", 32)
    monkeypatch.setattr("app.audio.MAX_MULTIPART_OVERHEAD", 128)
    parts = iter(
        [
            b'--limit\r\nContent-Disposition: form-data; name="audio"; filename="input.bin"\r\n\r\n',
            b"x" * 100,
            b"\r\n--limit--\r\n",
        ]
    )
    result = client.post(
        "/v1/classifications", headers={"Content-Type": "multipart/form-data; boundary=limit"}, content=parts
    )
    assert "content-length" not in result.request.headers
    assert result.status_code == 413
    assert result.json() == {"detail": "payload_too_large"}
    asr.transcribe.assert_not_called()


def test_asr_startup_failure_keeps_text_available(audio_settings, monkeypatch):
    monkeypatch.setattr("app.main.load_stt", Mock(side_effect=RuntimeError("сбой загрузки ASR")))
    with TestClient(create_app(audio_settings)) as client:
        assert client.get("/healthcheck").status_code == 200
        assert client.post("/predict", json={"text": TRANSCRIPT}).status_code == 200
        for endpoint in ("/v1/transcriptions", "/v1/classifications"):
            response = client.post(endpoint, files=UPLOAD)
            assert response.status_code == 503
            assert response.json() == {"detail": "asr_unavailable"}


def test_transcription_available_without_classifier(audio_settings, tmp_path, monkeypatch):
    asr = Mock()
    asr.transcribe.return_value = ASRResult("ok", None, TRANSCRIPT, "ru")
    monkeypatch.setattr("app.main.load_stt", lambda _: asr)
    with TestClient(create_app(replace(audio_settings, model_path=tmp_path / "missing.joblib"))) as client:
        assert client.post("/v1/classifications", files=UPLOAD).status_code == 503
        asr.transcribe.assert_not_called()
        assert client.post("/v1/transcriptions", files=UPLOAD).json()["transcript"] == TRANSCRIPT


def test_next_audio_request_runs_after_asr_failure(audio_client):
    client, asr = audio_client
    asr.transcribe.side_effect = [RuntimeError("сбой ASR"), ASRResult("ok", None, TRANSCRIPT, "ru")]
    assert client.post("/v1/transcriptions", files=UPLOAD).status_code == 503
    response = client.post("/v1/transcriptions", files=UPLOAD)
    assert response.status_code == 200
    assert response.json()["transcript"] == TRANSCRIPT
