import argparse
import csv
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, f1_score, accuracy_score)
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from data import load_data, prepare_data
from models import EXPERIMENTS, build_model
from storage import save_model, optimize_model

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def unpack(records):
    return [r[1] for r in records], [r[2] for r in records]


def evaluate(model, records, names, output, title):
    output.mkdir(parents=True, exist_ok=True)
    texts, labels = unpack(records)
    predictions = model.predict(texts)
    matrix = confusion_matrix(labels, predictions, labels=range(len(names)))
    report = classification_report(labels, predictions, labels=range(len(names)),
                                   target_names=names, output_dict=True, zero_division=0)
    save_json(output / "metrics.json", {
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "accuracy": accuracy_score(labels, predictions),
        "class_order": names, "confusion_matrix": matrix.tolist(), "report": report,
    })
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(matrix, display_labels=[n.rsplit(".", 1)[-1] for n in names]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    errors = [{"source_index": int(row[0]), "true": names[row[2]],
               "predicted": names[int(pred)], "text": row[1]}
              for row, pred in zip(records, predictions) if row[2] != pred]
    save_csv(output / "errors.csv", errors, ["source_index", "true", "predicted", "text"])

    guns, misc = names.index("talk.politics.guns"), names.index("talk.politics.misc")
    return {
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "accuracy": float(accuracy_score(labels, predictions)),
        "features": len(model.named_steps["tfidf"].vocabulary_),
        "guns_misc_errors": int(matrix[guns, misc] + matrix[misc, guns]),
    }


def validate_preparation():

    rows = [(0, "", 0), (1, "same", 0), (2, "same", 0),
            (3, "conflict", 0), (4, "conflict", 1), (5, "overlap", 1)]
    cleaned, stats = prepare_data(rows, {"overlap"})
    assert cleaned == [(1, "same", 0)]
    assert stats["empty"] == 1 and stats["duplicates_removed"] == 1
    assert stats["conflicting_rows_removed"] == 2 and stats["train_overlap_removed"] == 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--final", action="store_true")
    mode.add_argument("--optimize", action="store_true")
    args = parser.parse_args()

    if args.optimize:
        path = ROOT / "artifacts" / "model.joblib"
        if not path.exists():
            parser.error("Сначала выполните --final для создания модели.")
        raw_train, _ = load_data("train")
        train, _ = prepare_data(raw_train)
        raw_test, _ = load_data("test")
        test, _ = prepare_data(raw_test, {r[1] for r in raw_train})
        optimize_model(path, train + test)
        return

    validate_preparation()
    RESULTS.mkdir(exist_ok=True)
    print("Загрузка обучающих данных...", flush=True)
    raw, names = load_data("train")
    cleaned, stats = prepare_data(raw)
    print(f"Сообщения: {stats['raw']}; пустые: {stats['empty']}; "
          f"повторы: {stats['duplicates_removed']}; осталось: {len(cleaned)}", flush=True)
    fingerprint = hashlib.sha256(json.dumps(cleaned, ensure_ascii=False).encode()).hexdigest()
    if args.final:
        selection_path = RESULTS / "selection.json"
        if not selection_path.exists():
            parser.error("Сначала запустите эксперименты без --final.")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection["train_fingerprint"] != fingerprint:
            parser.error("Данные изменились: повторите валидационные эксперименты.")
        config = selection["config"]
        print(f"Выбор зафиксирован по валидации: {config['name']}", flush=True)
        final_model = build_model(config)
        final_model.fit(*unpack(cleaned))

        test_raw, test_names = load_data("test")
        assert names == test_names
        test, test_stats = prepare_data(test_raw, {r[1] for r in raw})
        assert not ({r[1] for r in test} & {r[1] for r in raw})
        save_json(RESULTS / "test_data_stats.json", test_stats)
        metrics = evaluate(final_model, test, names, RESULTS / "final", "Final test")
        save_json(RESULTS / "final" / "configuration.json", selection)
        artifacts = ROOT / "artifacts"
        artifacts.mkdir(exist_ok=True)
        save_model({"model": final_model, "class_names": names, "config": config},
                    artifacts / "model.joblib")

        restored = joblib.load(artifacts / "model.joblib")
        sample = [r[1] for r in test[:10]]
        assert np.array_equal(final_model.predict(sample), restored["model"].predict(sample))
        print(f"Финальный test macro-F1: {metrics['macro_f1']:.4f}; n={len(test)}", flush=True)
        return

    train, validation = train_test_split(cleaned, test_size=0.2, random_state=42,
                                        stratify=[r[2] for r in cleaned])
    assert not ({r[1] for r in train} & {r[1] for r in validation})
    save_json(RESULTS / "data_stats.json", stats)
    save_json(RESULTS / "split.json", {
        "seed": 42, "train_fingerprint": fingerprint, "class_names": names,
        "train_indices": [r[0] for r in train],
        "validation_indices": [r[0] for r in validation],
    })
    save_json(RESULTS / "environment.json", {
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "numpy": np.__version__, "threads": 1,
    })
    print(f"Обучение: {len(train)}; валидация: {len(validation)}", flush=True)
    summaries = []
    for config in EXPERIMENTS:
        print(f"Обучаем {config['name']}...", flush=True)
        model = build_model(config)
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(*unpack(train))
        seconds = time.perf_counter() - start
        warning_messages = [str(w.message) for w in caught]
        metrics = evaluate(model, validation, names, RESULTS / config["name"], config["name"])
        save_json(RESULTS / config["name"] / "configuration.json", config)
        save_json(RESULTS / config["name"] / "warnings.json", warning_messages)
        row = {"name": config["name"], **metrics, "fit_seconds": round(seconds, 3),
               "warnings": len(warning_messages)}
        summaries.append(row)
        print(f"  macro-F1={row['macro_f1']:.4f}; признаков={row['features']}; "
              f"guns/misc={row['guns_misc_errors']}; {seconds:.1f} с", flush=True)
        save_csv(RESULTS / "experiments.csv", summaries, list(row))
    if any(row["warnings"] for row in summaries):
        raise RuntimeError("Есть предупреждения обучения: проверьте warnings.json до выбора модели.")
    best = max(summaries, key=lambda r: r["macro_f1"])
    config = next(c for c in EXPERIMENTS if c["name"] == best["name"])
    save_json(RESULTS / "selection.json", {
        "config": config, "validation_macro_f1": best["macro_f1"],
        "train_fingerprint": fingerprint, "criterion": "highest validation macro-F1",
    })
    print(f"Выбрано: {best['name']}. Результаты в {RESULTS}", flush=True)


if __name__ == "__main__":

    with threadpool_limits(limits=1):
        main()

