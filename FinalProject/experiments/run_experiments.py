"""Серия экспериментов итоговой защиты: E0–E4 + политика отказа + дрейф-тест.

Запуск из корня репозитория:
    venv\\Scripts\\python.exe FinalProject\\experiments\\run_experiments.py

Каждый прогон логируется в MLflow (sqlite FinalProject/experiments/mlflow.db):
параметры, метрики, матрица ошибок, отчёт классификации, версии и хеш данных.
Результаты сохраняются в FinalProject/experiments/results/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import sklearn

HERE = Path(__file__).resolve()
FINAL_DIR = HERE.parents[1]
ROOT = HERE.parents[2]
sys.path.insert(0, str(FINAL_DIR))

from demo import (  # noqa: E402
    CONFIGS,
    LABEL_NAMES,
    build_pipeline,
    evaluate,
    group_split,
    ks_drift,
    load_dataset,
    policy_metrics,
    psi,
    validate_dataset,
)
from demo.models import blend_proba, select_weights  # noqa: E402

SEED = 42


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def plot_confusion(cm: np.ndarray, title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    names = [LABEL_NAMES[i] for i in range(3)]
    ax.set_xticks(range(3), names)
    ax.set_yticks(range(3), names)
    ax.set_xlabel("pred")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] < cm.max() * 0.6 else "white")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_comparison(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["run"] for r in rows]
    macro = [r["test"]["macro_f1"] for r in rows]
    acc = [r["test"]["accuracy"] for r in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - 0.18, macro, 0.36, label="macro-F1")
    ax.bar(x + 0.18, acc, 0.36, label="accuracy")
    for xi, v in zip(x - 0.18, macro):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, names, rotation=15)
    ax.set_ylim(0, 1.0)
    ax.set_title("Сравнение экспериментов (групповой test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "Datasets" / "df.csv")
    ap.add_argument("--max-rows", type=int, default=60_000)
    ap.add_argument("--out", type=Path, default=FINAL_DIR / "experiments" / "results")
    ap.add_argument("--tracking-uri", default=None, help="MLflow tracking URI (по умолчанию sqlite в experiments/mlflow.db)")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tracking_uri = args.tracking_uri or (FINAL_DIR / "experiments" / "mlflow.db").resolve().as_uri().replace("file:///", "sqlite:///")

    t_start = time.perf_counter()
    print(f"[1/6] Загрузка и очистка {args.data} (выборка до {args.max_rows})...")
    raw, clean, y, load_report = load_dataset(args.data, max_rows=args.max_rows, seed=SEED)
    validation = validate_dataset(clean, y)
    print(f"      строк после очистки: {len(clean)}; пропуски: {load_report.as_dict()['skipped']}")
    print(f"      валидация: {validation['class_counts']}; проблемы: {validation['issues'] or 'нет'}")

    fit_idx, dev_idx, test_idx = group_split(clean, y, seed=SEED)
    X_fit = [clean[i] for i in fit_idx]
    X_dev = [clean[i] for i in dev_idx]
    X_test = [clean[i] for i in test_idx]
    y_fit, y_dev, y_test = y[fit_idx], y[dev_idx], y[test_idx]
    split_info = {"fit": len(fit_idx), "dev": len(dev_idx), "test": len(test_idx), "seed": SEED}
    print(f"[2/6] Групповой сплит: {split_info}")

    data_sha = sha256_of(args.data)
    env = {
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset_sha256": data_sha,
        "normalizer": "demo-clean-v1",
    }

    use_mlflow = not args.no_mlflow
    if use_mlflow:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        exp_name = "final-defense-spam-clickbait"
        if mlflow.get_experiment_by_name(exp_name) is None:
            mlflow.create_experiment(
                exp_name,
                artifact_location=(FINAL_DIR / "experiments" / "mlartifacts").resolve().as_uri(),
            )
        mlflow.set_experiment(exp_name)

    results: list[dict] = []
    dev_probas: list[np.ndarray] = []
    test_probas: list[np.ndarray] = []

    def log_run(name: str, params: dict, metrics: dict, artifacts: dict[str, Path]) -> None:
        if not use_mlflow:
            return
        import mlflow

        with mlflow.start_run(run_name=name):
            mlflow.log_params({**params, "seed": SEED, "dataset_sha256": data_sha[:16], "sklearn": sklearn.__version__})
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            for p in artifacts.values():
                mlflow.log_artifact(str(p))

    # E0: мажоритарный baseline
    maj = int(np.bincount(y_fit).argmax())
    m0 = evaluate(y_test, np.full(len(y_test), maj))
    results.append({"run": "E0_majority", "params": {"strategy": "majority_class"}, "test": m0})
    log_run("E0_majority", {"strategy": "majority_class"}, {"test_macro_f1": m0["macro_f1"], "test_accuracy": m0["accuracy"]}, {})
    print(f"[3/6] E0 majority: macro-F1={m0['macro_f1']}, acc={m0['accuracy']}")

    # E1–E3: TF-IDF + LR
    for i, cfg_name in enumerate(CONFIGS, start=1):
        name = f"E{i}_{cfg_name}"
        print(f"[3/6] Обучение {name} на {len(X_fit)} строках...")
        model = build_pipeline(cfg_name)
        t0 = time.perf_counter()
        model.fit(X_fit, y_fit)
        fit_time = time.perf_counter() - t0
        p_dev = model.predict_proba(X_dev)
        p_test = model.predict_proba(X_test)
        dev_probas.append(p_dev)
        test_probas.append(p_test)
        m_dev = evaluate(y_dev, p_dev.argmax(axis=1))
        m_test = evaluate(y_test, p_test.argmax(axis=1))
        pol = policy_metrics(y_test, p_test)
        cm_path = args.out / f"cm_{name}.png"
        plot_confusion(np.array(m_test["confusion_matrix"]), name, cm_path)
        row = {
            "run": name,
            "params": {**CONFIGS[cfg_name], "fit_time_sec": round(fit_time, 1)},
            "dev": m_dev,
            "test": m_test,
            "policy_test": pol,
        }
        results.append(row)
        log_run(
            name,
            {"config": cfg_name, "n_fit": len(X_fit), "fit_time_sec": round(fit_time, 1)},
            {
                "dev_macro_f1": m_dev["macro_f1"],
                "test_macro_f1": m_test["macro_f1"],
                "test_accuracy": m_test["accuracy"],
                "test_f1_clickbait": m_test["per_class"]["clickbait"]["f1"],
                "test_f1_spam": m_test["per_class"]["spam"]["f1"],
                "coverage": pol["coverage"],
                "accuracy_classified": pol["accuracy_classified"],
            },
            {"confusion_matrix": cm_path},
        )
        print(f"      {name}: dev macro-F1={m_dev['macro_f1']}, test macro-F1={m_test['macro_f1']}, acc={m_test['accuracy']} ({fit_time:.0f} с)")

    # E4: ансамбль-смесь E1–E3, веса по dev
    print("[4/6] E4 blend: подбор весов на dev...")
    weights, dev_f1 = select_weights(dev_probas, y_dev)
    p_test_blend = blend_proba(test_probas, weights)
    m4 = evaluate(y_test, p_test_blend.argmax(axis=1))
    pol4 = policy_metrics(y_test, p_test_blend)
    cm4 = args.out / "cm_E4_blend.png"
    plot_confusion(np.array(m4["confusion_matrix"]), "E4_blend", cm4)
    results.append(
        {
            "run": "E4_blend",
            "params": {"weights_E1_E2_E3": weights, "selection": "dev macro-F1, шаг 0.1"},
            "dev": {"macro_f1": round(dev_f1, 4)},
            "test": m4,
            "policy_test": pol4,
        }
    )
    log_run(
        "E4_blend",
        {"weights": str(weights), "selection_split": "dev"},
        {"dev_macro_f1": dev_f1, "test_macro_f1": m4["macro_f1"], "test_accuracy": m4["accuracy"], "coverage": pol4["coverage"]},
        {"confusion_matrix": cm4},
    )
    print(f"      E4 weights={weights}: test macro-F1={m4['macro_f1']}, acc={m4['accuracy']}")

    # Политика отказа: sweep порогов для лучшей одиночной модели (E3)
    print("[5/6] Sweep порогов политики отказа (E3, test)...")
    p_best = test_probas[2]
    sweep = []
    for tau in np.arange(0.40, 0.85, 0.05):
        pm = policy_metrics(y_test, p_best, tau=float(tau), tau_spam=float(tau) + 0.10)
        sweep.append({"tau": round(float(tau), 2), "tau_spam": round(float(tau) + 0.10, 2), **pm})
    import csv as _csv

    with (args.out / "policy_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(sweep[0]))
        w.writeheader()
        w.writerows(sweep)

    # Дрейф-тест: fit vs test и симуляция «коротких голосовых»
    print("[6/6] Дрейф-тест (PSI/KS) и симуляция коротких реплик...")
    len_fit = np.array([len(t.split()) for t in X_fit])
    len_test = np.array([len(t.split()) for t in X_test])
    short_test = [" ".join(t.split()[:8]) for t in X_test]
    len_short = np.array([len(t.split()) for t in short_test])
    # model — обученная E3 (union_weighted, последняя в цикле CONFIGS); дообучение не требуется
    p_short = model.predict_proba(short_test)
    m_short = evaluate(y_test, p_short.argmax(axis=1))
    drift = {
        "fit_vs_test": {"psi_words": round(psi(len_fit, len_test), 4), **ks_drift(len_fit, len_test)},
        "simulated_short_8_words": {
            "psi_words": round(psi(len_fit, len_short), 4),
            "macro_f1_before": results[3]["test"]["macro_f1"],
            "macro_f1_after": m_short["macro_f1"],
            "delta_macro_f1": round(m_short["macro_f1"] - results[3]["test"]["macro_f1"], 4),
            "f1_clickbait_after": m_short["per_class"]["clickbait"]["f1"],
            "f1_spam_after": m_short["per_class"]["spam"]["f1"],
        },
    }
    print(f"      PSI fit/test={drift['fit_vs_test']['psi_words']}; короткие реплики: {drift['simulated_short_8_words']}")

    # Сохранение результатов
    plot_comparison(results, args.out / "experiments_comparison.png")
    import pandas as pd

    pd.DataFrame(
        [
            {
                "run": r["run"],
                "dev_macro_f1": r.get("dev", {}).get("macro_f1"),
                "test_macro_f1": r["test"]["macro_f1"],
                "test_accuracy": r["test"]["accuracy"],
                "test_f1_normal": r["test"]["per_class"]["normal"]["f1"],
                "test_f1_spam": r["test"]["per_class"]["spam"]["f1"],
                "test_f1_clickbait": r["test"]["per_class"]["clickbait"]["f1"],
                "coverage": r.get("policy_test", {}).get("coverage"),
                "accuracy_classified": r.get("policy_test", {}).get("accuracy_classified"),
            }
            for r in results
        ]
    ).to_csv(args.out / "experiments_comparison.csv", index=False)

    payload = {
        "env": env,
        "load_report": load_report.as_dict(),
        "validation": validation,
        "split": split_info,
        "runs": results,
        "policy_sweep": sweep,
        "drift": drift,
        "total_time_sec": round(time.perf_counter() - t_start, 1),
    }
    (args.out / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nГотово за {payload['total_time_sec']} с. Результаты: {args.out}")
    if use_mlflow:
        print(f"MLflow: {tracking_uri}  (mlflow ui --backend-store-uri {tracking_uri})")


if __name__ == "__main__":
    main()