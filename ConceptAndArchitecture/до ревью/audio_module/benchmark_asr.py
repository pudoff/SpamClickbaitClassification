from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path


def normalize(s: str) -> list[str]:
    s = s.lower().replace("ё", "е")
    return re.sub(r"[^а-яa-z0-9\s]", " ", s).split()


def wer(ref: str, hyp: str) -> float:
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1] / len(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--refs")
    ap.add_argument("--backends", default="gigaam,whisper")
    ap.add_argument("--out", default="asr_benchmark.csv")
    a = ap.parse_args()

    from .stt_service import STTService
    refs = {}
    if a.refs:
        with open(a.refs, encoding="utf-8") as f:
            refs = {r[0]: r[1] for r in csv.reader(f, delimiter="\t") if len(r) >= 2}
    files = sorted(p for p in Path(a.audio_dir).iterdir()
                   if p.suffix.lower() in {".ogg", ".oga", ".opus", ".wav", ".mp3", ".m4a"})

    rows, summary = [], []
    for b in a.backends.split(","):
        svc = STTService(b.strip())
        wers, t_tot, a_tot = [], 0.0, 0.0
        for p in files:
            t0 = time.perf_counter()
            r = svc.transcribe(p)
            dt = time.perf_counter() - t0
            w = wer(refs[p.name], r.text) if p.name in refs and r.status == "ok" else None
            if w is not None:
                wers.append(w)
            t_tot, a_tot = t_tot + dt, a_tot + r.duration_s
            rows.append([svc.backend.name, p.name, r.status, r.reason, round(r.duration_s, 1),
                         round(dt, 2), None if w is None else round(w, 4), r.text])
        summary.append((svc.backend.name, len(files),
                        sum(wers) / len(wers) if wers else None, t_tot / max(a_tot, 1e-9)))

    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w_ = csv.writer(f)
        w_.writerow(["model", "file", "status", "reason", "audio_s", "time_s", "wer", "text"])
        w_.writerows(rows)
    print("| модель | файлов | WER (средн.) | RTF |\n|---|---|---|---|")
    for name, n, w, rtf in summary:
        print(f"| {name} | {n} | {'—' if w is None else f'{w:.1%}'} | {rtf:.2f} |")


if __name__ == "__main__":
    main()
