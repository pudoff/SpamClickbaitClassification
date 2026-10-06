from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark_asr import wer
from .tts_generator import TTSAugmenter, sanitize_text

TEXT_COL, LABEL_COL = "text", "is_spam1_or_clickbait2"
LABELS = {0: "normal", 1: "spam", 2: "clickbait"}


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
    a = ap.parse_args()

    out = Path(a.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.jsonl"
    done = {json.loads(l)["id"] for l in meta_path.open(encoding="utf-8")} if meta_path.exists() else set()

    df = pd.read_csv(a.csv)
    df["clean"] = df[TEXT_COL].astype(str).map(sanitize_text)
    nw = df["clean"].str.split().str.len()
    df = df[(nw >= a.min_words) & (nw <= a.max_words)].drop_duplicates("clean")
    rng = np.random.default_rng(a.seed)

    tts = TTSAugmenter()
    asr = None
    if a.transcribe:
        from .stt_service import STTService
        asr = STTService(a.asr)

    with meta_path.open("a", encoding="utf-8") as f:
        for label, name in LABELS.items():
            part = df[df[LABEL_COL] == label]
            part = part.sample(min(a.per_class, len(part)), random_state=a.seed)
            for row_idx, row in part.iterrows():
                speaker = str(rng.choice(tts.backend.speakers))
                uid = hashlib.sha1(f"{row['clean']}|{speaker}".encode()).hexdigest()[:12]
                if uid in done:
                    continue
                path = out / "audio" / f"{uid}.ogg"
                try:
                    rec = tts.synthesize_augmented(row[TEXT_COL], path, speaker, rng)
                except Exception as e:  # плохой текст не должен ронять генерацию
                    print(f"skip {row_idx}: {e}")
                    continue
                rec.update(id=uid, file=f"audio/{uid}.ogg", label=int(label), label_name=name,
                           source_row=int(row_idx), split="train", synthetic=True)
                if asr is not None:
                    r = asr.transcribe(path)
                    rec.update(asr_text=r.text, asr_model=r.asr_model, asr_status=r.status,
                               wer=round(wer(rec["tts_text"], r.text), 4) if r.status == "ok" else None)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
    print("готово:", meta_path)


if __name__ == "__main__":
    main()
