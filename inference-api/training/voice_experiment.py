"""E-2a/E-5: сравнение TTS-обучения и проверка калибровки на реальных голосовых."""

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Self

import joblib
import numpy as np
import sklearn
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import f1_score

from app.artifacts import sha256_of
from app.audio import MAX_AUDIO_BYTES, load_stt
from app.config import DEFAULT_TAU, DEFAULT_TAU_SPAM, DEFAULT_TORCH_THREADS
from app.labels import CLASS_IDS, LABELS
from app.model import decide
from app.text import NORMALIZER_V2, get_normalizer
from audio_module.benchmark_asr import normalize as tokenize
from audio_module.benchmark_asr import wer
from audio_module.tts_generator import sanitize_text
from training.metrics import evaluate
from training.train import CONFIGS, RANDOM_STATE, build_pipeline

VARIANTS = ("text", "tts_asr", "mixed")
REFUSAL = -1
SPAM = 1
TARGET_PRECISION = 0.85
normalize = get_normalizer(NORMALIZER_V2)


class VoiceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    id: str = Field(min_length=1)
    file: str = Field(min_length=1)
    speaker: str = Field(min_length=1)
    group: str = Field(min_length=1)
    split: Literal["voice-calib", "voice-holdout"]
    synthetic: Literal[False]
    consent: Literal[True]
    label: Literal[0, 1, 2] | None
    expected_reason: Literal["no_speech", "speech_too_short", "unsupported_language"] | None = None
    reference: str | None = None

    @model_validator(mode="after")
    def validate_target(self) -> Self:
        if self.label is None:
            if self.expected_reason is None or self.split == "voice-calib":
                raise ValueError("ожидаемые отказы с причиной включаются только в voice-holdout")
        elif self.expected_reason is not None:
            raise ValueError("причина отказа допустима только при label=null")
        if self.reference is not None and not tokenize(self.reference):
            raise ValueError("эталонная расшифровка должна содержать слова")
        return self


def jsonl(path: Path):
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield number, json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path.name}, строка {number}: некорректный JSON") from exc


def load_pairs(path: Path, tts_model: str):
    source, transcript, labels = [], [], []
    seen = set()
    for number, row in jsonl(path):
        if row.get("tts_model") != tts_model:
            continue
        if row.get("synthetic") is not True or row.get("split") != "train":
            raise ValueError(f"строка TTS {number}: требуется синтетическая запись из train")
        if row.get("asr_status") != "ok":
            continue
        if row.get("asr_model") != "gigaam-v3-e2e-rnnt":
            raise ValueError(f"строка TTS {number}: нужна расшифровка GigaAM v3 e2e RNNT")
        if (
            type(row.get("label")) is not int
            or row["label"] not in CLASS_IDS
            or not isinstance(row.get("source_text"), str)
            or not isinstance(row.get("asr_text"), str)
        ):
            raise ValueError(f"строка TTS {number}: нужна метка класса и исходный текст")
        original = normalize(row["source_text"])
        recognized = normalize(row["asr_text"])
        if not original or not recognized:
            continue
        if original in seen:
            continue
        seen.add(original)
        source.append(original)
        transcript.append(recognized)
        labels.append(row["label"])
    if set(labels) != set(CLASS_IDS):
        raise ValueError("нужны успешные расшифровки TTS для каждого из трёх классов")
    return source, transcript, np.asarray(labels)


def load_voice(path: Path, train_texts: set[str]):
    root = path.parent.resolve()
    seen = {"id": {}, "audio": {}, "speaker": {}, "group": {}, "reference": {}}
    spoken_train = {sanitize_text(text).casefold() for text in train_texts}
    splits = {"voice-calib": [], "voice-holdout": []}
    for number, row in jsonl(path):
        try:
            record = VoiceRecord.model_validate(row)
        except ValueError as exc:
            raise ValueError(f"строка голосового манифеста {number}: неверная схема") from exc
        audio = (root / record.file).resolve()
        if Path(record.file).is_absolute() or not audio.is_relative_to(root) or not audio.is_file():
            raise ValueError(f"строка голосового манифеста {number}: нужен локальный файл рядом с манифестом")
        reference = normalize(record.reference or "")
        if audio.stat().st_size > MAX_AUDIO_BYTES:
            raise ValueError(f"строка голосового манифеста {number}: файл больше лимита API")
        spoken = sanitize_text(record.reference or "").casefold()
        if (reference and reference in train_texts) or (spoken and spoken in spoken_train):
            raise ValueError(f"строка голосового манифеста {number}: эталон пересекается с train")
        for key, value in (
            ("id", record.id),
            ("audio", sha256_of(audio)),
            ("speaker", record.speaker),
            ("group", record.group),
            ("reference", reference),
        ):
            if not value:
                continue
            previous = seen[key].get(value)
            if previous is not None and (previous != record.split or key in {"id", "audio"}):
                raise ValueError(f"строка голосового манифеста {number}: пересечение или дубликат по {key}")
            seen[key][value] = record.split
        splits[record.split].append((record, audio))
    for split, rows in splits.items():
        if {record.label for record, _ in rows if record.label is not None} != set(CLASS_IDS):
            raise ValueError(f"{split}: нужны записи всех трёх классов")
    return splits["voice-calib"], splits["voice-holdout"]


def recognize(stt, rows):
    return [stt.transcribe(path) for _, path in rows]


def scores_for(model, results):
    recognized = [i for i, result in enumerate(results) if result.text and result.status == "ok"]
    scores = np.zeros((len(results), len(CLASS_IDS)))
    if recognized:
        scores[recognized] = model.predict_proba([normalize(results[i].text) for i in recognized])
    return scores


def expected_labels(rows):
    return np.asarray([REFUSAL if record.label is None else record.label for record, _ in rows])


def select_policy(scores, truth):
    # Для normal и clickbait сохраняем минимальную уверенность, принятую в сервисе.
    confidence = scores.max(axis=1)
    spam = scores.argmax(axis=1) == SPAM
    order = np.flatnonzero(spam)[np.argsort(-confidence[spam], kind="stable")]
    if not len(order):
        return None
    counts = np.arange(1, len(order) + 1)
    best = np.flatnonzero(
        (np.cumsum(truth[order] == SPAM) / counts >= TARGET_PRECISION)
        & (np.r_[confidence[order][:-1] != confidence[order][1:], True])
    )
    if not len(best):
        return None
    return {"tau": DEFAULT_TAU, "tau_spam": float(confidence[order[best[-1]]])}


def decisions(scores, policy):
    predictions = []
    for probabilities in scores:
        label = None
        if probabilities.any():
            decision = decide(
                dict(zip(LABELS.values(), probabilities, strict=True)),
                policy["tau"],
                policy["tau_spam"],
                None,
            )
            label = decision.label_id
        predictions.append(REFUSAL if label is None else label)
    return np.asarray(predictions)


def macro_f1(truth, predicted):
    accepted = (truth != REFUSAL) & (predicted != REFUSAL)
    if not accepted.any():
        return 0.0
    return float(f1_score(truth[accepted], predicted[accepted], labels=CLASS_IDS, average="macro", zero_division=0))


def ece(scores, truth):
    valid = scores.any(axis=1) & (truth != REFUSAL)
    if not valid.any():
        return None
    confidence = scores[valid].max(axis=1)
    correct = scores[valid].argmax(axis=1) == truth[valid]
    bins = np.minimum((confidence * 10).astype(int), 9)
    return float(np.abs(np.bincount(bins, weights=correct - confidence)).sum() / len(bins))


def measure(scores, truth, predicted):
    eligible = truth != REFUSAL
    classified = eligible & (predicted != REFUSAL)
    spam = predicted == SPAM
    return {
        "f1_macro_classified": macro_f1(truth, predicted),
        "classified": int(classified.sum()),
        "coverage": float(classified.sum() / eligible.sum()),
        "precision_spam": float((spam & (truth == SPAM)).sum() / max(spam.sum(), 1)),
        "recall_spam": float((spam & (truth == SPAM)).sum() / (truth == SPAM).sum()),
        "fpr_normal_to_spam": float(spam[truth == 0].mean()),
        "ece": ece(scores, truth),
        "class_metrics_classified": evaluate(truth[classified], predicted[classified]) if classified.any() else None,
    }


def paired_interval(rows, truth, baseline, variant):
    speakers = {}
    for i, (record, _) in enumerate(rows):
        speakers.setdefault(record.speaker, []).append(i)
    groups = list(speakers.values())
    if len(groups) < 2:
        return None
    rng = np.random.default_rng(RANDOM_STATE)
    differences = []
    for _ in range(1000):
        idx = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
        differences.append(macro_f1(truth[idx], variant[idx]) - macro_f1(truth[idx], baseline[idx]))
    return np.quantile(differences, [0.025, 0.975]).tolist()


def asr_metrics(rows, results):
    rates, counts = {}, Counter()
    word_errors, words, references = 0.0, 0, 0
    for (record, _), result in zip(rows, results, strict=True):
        if record.expected_reason:
            counts[record.expected_reason] += 1
            rates[record.expected_reason] = rates.get(record.expected_reason, 0) + (
                result.reason == record.expected_reason
            )
        elif record.reference is not None:
            reference_words = len(tokenize(record.reference))
            references += 1
            words += reference_words
            word_errors += reference_words * wer(record.reference, result.text)
    return {
        "wer": word_errors / words if words else None,
        "wer_reference_count": references,
        "expected_refusals": {reason: {"n": n, "correct": rates[reason]} for reason, n in counts.items()},
    }


def run(args):
    original, synthetic, labels = load_pairs(args.tts_meta, args.tts_model)
    variants = dict(zip(VARIANTS, (original, synthetic, original + synthetic), strict=True))
    targets = dict(zip(VARIANTS, (labels, labels, np.tile(labels, 2)), strict=True))
    report = {
        "config": args.config,
        "tts_model": args.tts_model,
        "tts_meta_sha256": sha256_of(args.tts_meta),
        "paired_train_counts": dict(Counter(labels.tolist())),
        "train_sizes": {name: len(texts) for name, texts in variants.items()},
    }
    if args.voice_manifest:
        calib, holdout = load_voice(args.voice_manifest, set(original) | set(synthetic))
    models = {
        name: build_pipeline(CONFIGS[args.config], targets[name]).fit(texts, targets[name])
        for name, texts in variants.items()
    }
    if not args.voice_manifest:
        return {**report, "status": "training_only"}, None, None

    stt = load_stt(DEFAULT_TORCH_THREADS)
    calib_results = recognize(stt, calib)
    calib_truth = expected_labels(calib)
    valid = np.asarray([result.status == "ok" and bool(result.text) for result in calib_results])
    if np.bincount(calib_truth[valid], minlength=len(CLASS_IDS)).min() < 2:
        raise ValueError("после ASR в voice-calib нужны хотя бы две записи каждого класса")
    calib_texts = [normalize(result.text) for result in calib_results if result.status == "ok" and result.text]
    policies, calibration = {}, {}
    for name, model in models.items():
        before = scores_for(model, calib_results)
        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid", cv=2)
        calibrated.fit(calib_texts, calib_truth[valid])
        models[name] = calibrated
        after = scores_for(calibrated, calib_results)
        policies[name] = select_policy(after, calib_truth)
        policy = policies[name] or {"tau": DEFAULT_TAU, "tau_spam": DEFAULT_TAU_SPAM}
        calibration[name] = {
            **measure(after, calib_truth, decisions(after, policy)),
            "ece_before": ece(before, calib_truth),
            "tuned_policy": policies[name],
        }
    candidate = max(
        (name for name in VARIANTS if policies[name] is not None),
        key=lambda name: (calibration[name]["f1_macro_classified"], calibration[name]["coverage"]),
        default=None,
    )

    holdout_results = recognize(stt, holdout)
    truth = expected_labels(holdout)
    metrics, predictions = {}, {}
    for name, model in models.items():
        # Общая политика нужна для парного сравнения влияния обучающих данных.
        scores = scores_for(model, holdout_results)
        predictions[name] = decisions(scores, {"tau": DEFAULT_TAU, "tau_spam": DEFAULT_TAU_SPAM})
        metrics[name] = measure(scores, truth, predictions[name])
    for name in VARIANTS[1:]:
        metrics[name]["delta_f1_macro"] = metrics[name]["f1_macro_classified"] - metrics["text"]["f1_macro_classified"]
        metrics[name]["delta_f1_macro_ci95"] = paired_interval(holdout, truth, predictions["text"], predictions[name])
    split_speakers = {
        name: len({record.speaker for record, _ in rows})
        for name, rows in (("voice-calib", calib), ("voice-holdout", holdout))
    }
    holdout_counts = Counter(truth.tolist())
    holdout_asr = asr_metrics(holdout, holdout_results)
    report.update(
        status="evaluated",
        voice_manifest_sha256=sha256_of(args.voice_manifest),
        split_sizes={"voice-calib": len(calib), "voice-holdout": len(holdout)},
        split_speakers=split_speakers,
        split_class_counts={
            "voice-calib": dict(Counter(calib_truth.tolist())),
            "voice-holdout": dict(holdout_counts),
        },
        sample_targets_met=(
            len(calib) >= 200
            and len(holdout) >= 400
            and min(holdout_counts[label] for label in CLASS_IDS) >= 100
            and holdout_counts[REFUSAL] >= 60
            and sum(split_speakers.values()) >= 12
            and holdout_asr["wer_reference_count"] >= 150
        ),
        holdout_asr=holdout_asr,
        calibration=calibration,
        holdout_comparison=metrics,
        candidate=candidate,
    )
    if candidate is None:
        return report, None, None
    policy = policies[candidate]
    scores = scores_for(models[candidate], holdout_results)
    report["candidate_holdout"] = measure(scores, truth, decisions(scores, policy))
    report["candidate_policy"] = policy
    return (
        report,
        models[candidate],
        {
            "name": "tfidf-logreg-voice",
            "version": args.version,
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "sklearn_version": sklearn.__version__,
            "config": {
                "normalizer": NORMALIZER_V2,
                "variant": candidate,
                "calibration": "sigmoid",
                "voice_policy": policy,
            },
            "config_name": args.config,
            "metrics": {"f1_macro": report["candidate_holdout"]["f1_macro_classified"], **report["candidate_holdout"]},
            "dataset": {"voice_manifest_sha256": report["voice_manifest_sha256"], **report["split_sizes"]},
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tts-meta", required=True, type=Path)
    parser.add_argument("--tts-model", choices=("piper", "silero_v4_ru"), default="piper")
    parser.add_argument("--voice-manifest", type=Path)
    parser.add_argument("--config", choices=sorted(CONFIGS), default="exp4")
    parser.add_argument("--version", default="1")
    parser.add_argument("--out", required=True, type=Path, help="новый каталог отчёта и кандидата")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out должен быть новым каталогом")
    report, model, meta = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent) as temp:
        output = Path(temp) / "result"
        output.mkdir()
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if model is not None:
            joblib.dump(model, output / "model.joblib", compress=3)
            (output / "model_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        output.rename(args.out)
    print(f"{report['status']}: {args.out / 'report.json'}")


if __name__ == "__main__":
    main()
