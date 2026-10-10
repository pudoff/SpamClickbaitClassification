"""Единые пути для данных, учебного кода и результатов экспериментов."""

from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = DATASET_DIR / "scripts"
NOTEBOOK_DIR = DATASET_DIR / "notebooks"
OUTPUT_DIR = DATASET_DIR / "outputs"
EDA_OUTPUT_DIR = OUTPUT_DIR / "eda"
