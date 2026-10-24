"""Снимки для характеризационных тестов: tests/snapshots/<имя>.

Обновить снимки (только осознанно, при изменении контракта): UPDATE_SNAPSHOTS=1 pytest.
"""

import json
import os
from pathlib import Path

SNAPSHOTS = Path(__file__).parent / "snapshots"


def dump(data) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def snapshot(name: str, data) -> str:
    """Текст сохранённого снимка; с UPDATE_SNAPSHOTS=1 сначала перезаписывает его данными data."""
    path = SNAPSHOTS / name
    if os.getenv("UPDATE_SNAPSHOTS"):
        path.parent.mkdir(exist_ok=True)
        path.write_text(dump(data), encoding="utf-8")
    return path.read_text(encoding="utf-8")
