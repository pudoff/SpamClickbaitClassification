import sys
import tempfile
from pathlib import Path

import numpy as np

from .benchmark_asr import wer
from .stt_service import STTService
from .tts_generator import TTSAugmenter

TEXTS = {
    "spam": "Вы выиграли приз в нашей лотерее. Перейдите по ссылке и получите деньги на карту сегодня.",
    "clickbait": "Вы не поверите, что случилось с этой звездой после скандала. Врачи в шоке от такого результата.",
    "normal": "Привет, давай встретимся завтра в шесть часов у входа в кино, я возьму билеты заранее",
}

if __name__ == "__main__":
    tts, asr = TTSAugmenter(), STTService(sys.argv[1] if len(sys.argv) > 1 else "gigaam")
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        for label, text in TEXTS.items():
            meta = tts.synthesize_augmented(text, Path(tmp) / f"{label}.ogg", rng=rng)
            r = asr.transcribe(Path(tmp) / f"{label}.ogg")
            print(f"[{label}] status={r.status} WER={wer(meta['tts_text'], r.text):.1%} "
                  f"time={r.timings_ms.get('total')}ms aug=speed{meta['speed']}/snr{meta['snr_db']}/opus{meta['opus_kbps']}k")
            print("   ASR:", r.text)
