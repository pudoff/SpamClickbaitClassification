import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.model import load_model
from audio_module.schemas import ASRResult
from training import voice_experiment as experiment

PHRASES = ("купим молоко домой", "перейдите заберите выигрыш", "узнайте невероятную тайну")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def voice_inputs(tmp_path):
    prefixes = ("обычно", "сегодня", "сейчас", "теперь", "вместе")
    tts = tmp_path / "tts.jsonl"
    write_jsonl(
        tts,
        [
            {
                "label": label,
                "source_text": f"{prefix} {phrase}",
                "tts_model": "piper",
                "asr_status": "ok",
                "asr_model": "gigaam-v3-e2e-rnnt",
                "asr_text": f"{phrase} {prefix}",
                "split": "train",
                "synthetic": True,
            }
            for label, phrase in enumerate(PHRASES)
            for prefix in prefixes
        ],
    )
    records = []
    for split in ("voice-calib", "voice-holdout"):
        for index in range(12):
            label = index % 3
            record_id = f"{split}-{index}"
            file = tmp_path / f"{record_id}.ogg"
            file.write_text(record_id, encoding="ascii")
            records.append(
                {
                    "id": record_id,
                    "file": file.name,
                    "speaker": f"{split}-speaker-{index // 6}",
                    "group": f"{split}-group-{index % 3}",
                    "split": split,
                    "synthetic": False,
                    "consent": True,
                    "label": label,
                    "reference": f"{PHRASES[label]} {'утром' if split == 'voice-calib' else 'вечером'}",
                }
            )
    return tts, records


def test_experiment_calibrates_before_holdout_and_exports_loadable_candidate(tmp_path, voice_inputs, monkeypatch):
    tts, records = voice_inputs
    records[-1].pop("reference")
    manifest = tmp_path / "voice.jsonl"
    write_jsonl(manifest, records)
    by_file = {record["file"]: record for record in records}
    calls = []

    def transcribe(path):
        record = by_file[path.name]
        calls.append(record["split"])
        return ASRResult("ok", None, record.get("reference", PHRASES[record["label"]]))

    monkeypatch.setattr(experiment, "load_stt", lambda _: SimpleNamespace(transcribe=transcribe))
    out = tmp_path / "result"
    monkeypatch.setattr(
        "sys.argv",
        ["voice_experiment", "--tts-meta", str(tts), "--voice-manifest", str(manifest), "--out", str(out)],
    )
    experiment.main()
    assert calls == ["voice-calib"] * 12 + ["voice-holdout"] * 12
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert set(report["holdout_comparison"]) == set(experiment.VARIANTS)
    assert not report["sample_targets_met"]
    assert report["holdout_comparison"]["tts_asr"]["delta_f1_macro_ci95"] == [0, 0]
    assert report["holdout_asr"]["wer"] == 0
    assert report["holdout_asr"]["wer_reference_count"] == 11
    candidate = load_model(out / "model.joblib", out / "model_meta.json")
    assert candidate.voice_policy.model_dump() == report["candidate_policy"]
    assert sum(candidate.predict_proba("купим молоко вечером").values()) == pytest.approx(1)


@pytest.mark.parametrize(
    ("field", "updates", "error"),
    [
        ("speaker", {}, "speaker"),
        ("file", {}, "audio"),
        ("group", {}, "group"),
        (None, {"consent": False}, "схема"),
        (None, {"synthetic": True}, "схема"),
    ],
)
def test_manifest_rejects_leakage_and_unconsented_records(tmp_path: Path, voice_inputs, field, updates, error):
    _, records = voice_inputs
    if field is not None:
        records[-1][field] = records[0][field]
    records[-1].update(updates)
    manifest = tmp_path / "voice.jsonl"
    write_jsonl(manifest, records)
    with pytest.raises(ValueError, match=error):
        experiment.load_voice(manifest, set())


def test_policy_counts_equal_confidences_together_and_preserves_asr_refusal():
    scores = np.asarray([[0.1, 0.8, 0.1], [0.1, 0.7, 0.2], [0.2, 0.7, 0.1], [0, 0, 0]])
    truth = np.asarray([1, 1, 0, 2])
    policy = experiment.select_policy(scores, truth)
    assert policy["tau_spam"] == 0.8
    assert experiment.decisions(scores, policy).tolist() == [1, -1, -1, -1]
    assert experiment.select_policy(scores, np.zeros(4)) is None
