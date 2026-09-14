import csv
import json
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent


def save_model(bundle, path):

    report_path = ROOT / "results" / "compression.json"
    compression = ("xz", 6)
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        compression = tuple(report["selected_compression"])
    with TemporaryDirectory(dir=path.parent, prefix="save-") as directory:
        candidate = Path(directory) / "model.joblib"
        joblib.dump(bundle, candidate, compress=compression)
        restored = joblib.load(candidate)
        if joblib.hash(bundle) != joblib.hash(restored):
            raise RuntimeError("Состояние модели изменилось при сохранении.")
        candidate.replace(path)


def optimize_model(path, records):

    bundle = joblib.load(path)
    before_bytes = path.stat().st_size
    texts = [row[1] for row in records]
    labels = bundle["model"].predict(texts)
    probabilities = bundle["model"].predict_proba(texts)
    state_hash = joblib.hash(bundle)
    rows = []
    formats = [("none", 0), ("zlib_3", ("zlib", 3)),
               ("gzip_3", ("gzip", 3)), ("xz_3", ("xz", 3)), ("xz_6", ("xz", 6))]

    with TemporaryDirectory(dir=path.parent, prefix="compression-") as directory:
        for name, compression in formats:
            candidate = Path(directory) / f"{name}.joblib"
            start = perf_counter()
            joblib.dump(bundle, candidate, compress=compression)
            write_seconds = perf_counter() - start
            load_times = []
            for _ in range(3):
                start = perf_counter()
                restored = joblib.load(candidate)
                load_times.append(perf_counter() - start)
            same_state = joblib.hash(restored) == state_hash
            same_labels = np.array_equal(labels, restored["model"].predict(texts))
            same_probabilities = np.array_equal(probabilities, restored["model"].predict_proba(texts))
            if not (same_state and same_labels and same_probabilities):
                raise RuntimeError(f"Проверка без потерь не пройдена: {name}")
            rows.append({"format": name, "bytes": candidate.stat().st_size,
                         "write_seconds": round(write_seconds, 6),
                         "load_seconds_median_3": round(median(load_times), 6),
                         "same_state": same_state, "same_predictions": same_labels,
                         "same_probabilities": same_probabilities})
            print(f"{name}: {rows[-1]['bytes']} bytes; проверка пройдена", flush=True)
        best = min(rows, key=lambda row: row["bytes"])
        selected_compression = next(value for name, value in formats if name == best["format"])
        if best["bytes"] > before_bytes:
            raise RuntimeError("Ни один кандидат не меньше текущей модели; рабочий файл сохранён.")
        (Path(directory) / f"{best['format']}.joblib").replace(path)
    report = {
        "method": "lossless serialization compression, no retraining",
        "before_bytes": before_bytes, "after_bytes": path.stat().st_size,
        "selected_format": best["format"], "selected_compression": selected_compression,
        "verified_texts": len(texts), "original_state_hash": state_hash,
        "features": len(bundle["model"].named_steps["tfidf"].vocabulary_),
        "classifier_weight_shape": list(bundle["model"].named_steps["classifier"].feature_log_prob_.shape),
        "measurements": rows,
    }
    results = ROOT / "results"
    (results / "compression.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (results / "compression.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Размер: {before_bytes} → {report['after_bytes']} байт", flush=True)
