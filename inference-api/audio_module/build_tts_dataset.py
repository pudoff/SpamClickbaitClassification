from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from app.labels import LABELS
from training.data import load_fixed_csv

from .benchmark_asr import wer
from .tts_generator import PiperBackend, TTSAugmenter, sanitize_text

TEXT_COL, LABEL_COL = "text", "is_spam1_or_clickbait2"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="data/tts_dataset")
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--min-words", type=int, default=10)
    ap.add_argument("--max-words", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--transcribe", action="store_true", help="прогнать через ASR -> asr_text и wer")
    ap.add_argument("--asr", default="gigaam")
    ap.add_argument("--tts", choices=("silero", "piper"), default="silero")
    ap.add_argument("--piper-bin", default="piper")
    ap.add_argument("--piper-voice", action="append", default=[], metavar="ИМЯ=ПУТЬ.onnx")
    a = ap.parse_args()
    if a.per_class < 1 or not 1 <= a.min_words <= a.max_words:
        ap.error("нужны положительные размеры и min-words <= max-words")
    if a.piper_voice and a.tts != "piper":
        ap.error("--piper-voice требует --tts piper")
    backend = "silero"
    if a.tts == "piper":
        if not shutil.which(a.piper_bin) or not a.piper_voice:
            ap.error("укажите доступный --piper-bin и хотя бы один --piper-voice")
        voices = {}
        for item in a.piper_voice:
            name, _, filename = item.partition("=")
            path = Path(filename)
            if not name or name in voices or path.suffix != ".onnx" or not path.is_file():
                ap.error("каждый --piper-voice требует уникальное имя и существующий файл .onnx")
            if not path.with_suffix(".onnx.json").is_file():
                ap.error(f"для голоса {name} нужен конфиг {path.name}.json")
            voices[name] = str(path.resolve())
        backend = PiperBackend(voices, a.piper_bin)

    out = Path(a.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.jsonl"
    done = {}
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as f:
            done = {rec["id"]: rec for rec in map(json.loads, f)}

    texts, labels, skipped = load_fixed_csv(Path(a.csv))
    if skipped:
        print("пропущено строк:", dict(skipped))
    df = pd.DataFrame({TEXT_COL: texts, LABEL_COL: labels})
    df["clean"] = df[TEXT_COL].map(sanitize_text)
    nw = df["clean"].str.split().str.len()
    df = df[(nw >= a.min_words) & (nw <= a.max_words)].drop_duplicates("clean")
    rng = np.random.default_rng(a.seed)

    tts = TTSAugmenter(backend)
    asr = None
    if a.transcribe:
        from .stt_service import STTService

        asr = STTService(a.asr)

    ready = 0
    with meta_path.open("a", encoding="utf-8") as f:
        for label, name in LABELS.items():
            part = df[df[LABEL_COL] == label]
            part = part.sample(min(a.per_class, len(part)), random_state=a.seed)
            for row_idx, row in part.iterrows():
                speaker = str(rng.choice(tts.backend.speakers))
                seed = int(rng.integers(2**32))
                key = (row[TEXT_COL], label, tts.backend.name, speaker, seed, a.asr if asr else None)
                uid = hashlib.sha256(json.dumps(key, ensure_ascii=False).encode()).hexdigest()[:24]
                if uid in done:
                    if not (out / done[uid]["file"]).is_file():
                        raise ValueError("не найдено аудио из метаданных, используйте новый каталог --out")
                    ready += 1
                    continue
                path = out / "audio" / f"{uid}.ogg"
                try:
                    rec = tts.synthesize_augmented(row[TEXT_COL], path, speaker, np.random.default_rng(seed))
                except Exception as e:  # плохой текст не должен ронять генерацию
                    print(f"skip {row_idx}: {e}")
                    continue
                rec.update(
                    id=uid,
                    file=f"audio/{uid}.ogg",
                    label=int(label),
                    label_name=name,
                    source_row=int(row_idx),
                    source_text=row[TEXT_COL],
                    split="train",
                    synthetic=True,
                )
                if asr is not None:
                    r = asr.transcribe(path)
                    rec.update(
                        asr_text=r.text,
                        asr_model=r.asr_model,
                        asr_status=r.status,
                        wer=round(wer(rec["tts_text"], r.text), 4) if r.status == "ok" else None,
                    )
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                ready += 1
    if not ready:
        raise SystemExit("не удалось подготовить ни одной записи")
    print("готово:", meta_path)


if __name__ == "__main__":
    main()
