"""Проверить и записать результаты EDA-ячеек без запуска дорогого GridSearch.

Запуск: python Datasets/scripts/verify_eda.py. Требуется окружение ноутбука.
"""

import ast
import base64
from contextlib import redirect_stdout, redirect_stderr
from io import BytesIO, StringIO
import json
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_utils import TARGET_COLUMN, clean_text, load_dataset
from dataset_paths import NOTEBOOK_DIR, EDA_OUTPUT_DIR


def python_source(cell):
    return "".join(line for line in cell["source"] if not line.lstrip().startswith("%"))


def main():
    notebook_path = NOTEBOOK_DIR / "model.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse(python_source(cell), filename=f"cell_{index}")
    # Проверка восстановления разделителей, CSV-кавычек и диагностики меток.
    with tempfile.TemporaryDirectory(prefix="dataset_eda_") as temporary:
        fixture = Path(temporary) / "df.csv"
        fixture.write_text(
            f'text,{TARGET_COLUMN}\n"Текст, с ""кавычками""",0\n'
            'Заголовок;2\nПример,2000\nтекст,label\n,1\n\n', encoding="utf-8")
        sample, issues, stats = load_dataset(fixture)
        assert sample.iloc[0, 0] == 'Текст, с "кавычками"'
        assert sample[TARGET_COLUMN].tolist() == [0, 2, 2000, 1]
        assert set(issues["reason"]) == {"embedded_header", "invalid_label"}
        assert stats["blank_lines"] == 1
        assert clean_text("Привет <link> 123") == "привет url num"
    namespace = {"__name__": "__main__"}
    current_outputs = []

    def capture_display(*values, **kwargs):
        for value in values:
            data = {"text/plain": [repr(value)]}
            if hasattr(value, "_repr_html_"):
                rendered = value._repr_html_()
                if rendered:
                    data["text/html"] = [rendered]
            current_outputs.append({"output_type": "display_data", "data": data, "metadata": {}})

    def capture_show(*args, **kwargs):
        for number in plt.get_fignums():
            buffer = BytesIO()
            plt.figure(number).savefig(buffer, format="png", bbox_inches="tight")
            current_outputs.append({"output_type": "display_data", "data": {
                "image/png": base64.b64encode(buffer.getvalue()).decode("ascii"),
                "text/plain": [repr(plt.figure(number))]}, "metadata": {}})
            plt.close(number)

    plt.show = capture_show
    executed = []
    count = 0
    training_start = next(index for index, cell in enumerate(notebook["cells"])
                          if cell["cell_type"] == "markdown"
                          and "**7. Модель 1" in "".join(cell["source"]))
    for index, cell in enumerate(notebook["cells"]):
        if index >= training_start or cell["cell_type"] != "code":
            continue
        current_outputs.clear()
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(compile(python_source(cell), f"cell_{index}", "exec"), namespace)
        if index == 2:
            namespace["display"] = capture_display
        outputs = []
        if stdout.getvalue():
            outputs.append({"output_type": "stream", "name": "stdout", "text": stdout.getvalue().splitlines(keepends=True)})
        # Progress bars are omitted; warnings are retained for review in verification.json.
        outputs.extend(current_outputs.copy())
        if "profile.to_notebook_iframe()" in python_source(cell) and namespace.get("profile") is not None:
            from html import escape
            iframe = '<iframe width="100%" height="800" srcdoc="' + escape(namespace["profile"].to_html(), quote=True) + '"></iframe>'
            outputs.append({"output_type": "display_data", "data": {"text/html": [iframe]}, "metadata": {}})
        count += 1
        cell["execution_count"] = count
        cell["outputs"] = outputs
        executed.append({"cell": index, "stdout": stdout.getvalue(), "stderr": stderr.getvalue()})
        print(f"Cell {index}: OK", flush=True)
    prepared = namespace["df"]
    raw = namespace["df_raw"]
    assert len(raw) == 641738 and set(prepared[TARGET_COLUMN]) == {0, 1, 2}
    assert not prepared["model_text"].eq("").any()
    assert prepared["model_text"].is_unique
    assert set(namespace["X_train_clean"]).isdisjoint(set(namespace["X_test_clean"]))
    result = {
        "checked_all_code_cells_syntax": True, "loader_fixture_passed": True,
        "executed_cells": executed, "model_training_executed": False,
        "raw_rows": len(raw), "prepared_rows": len(prepared),
        "prepared_class_counts": {str(k): int(v) for k, v in prepared[TARGET_COLUMN].value_counts().sort_index().items()},
        "train_rows": len(namespace["X_train"]), "test_rows": len(namespace["X_test"]),
        "train_test_clean_text_overlap": 0,
    }
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (EDA_OUTPUT_DIR / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "executed_cells"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
