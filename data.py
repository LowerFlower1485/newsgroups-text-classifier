from collections import Counter
from statistics import median
from typing import cast

from sklearn.datasets import fetch_20newsgroups
from sklearn.utils import Bunch

CATEGORIES = ["talk.politics.guns", "talk.politics.mideast", "talk.politics.misc"]


def load_data(subset):
    dataset = cast(Bunch, fetch_20newsgroups(
        subset=subset, categories=CATEGORIES,
        remove=("headers", "footers", "quotes"),
        shuffle=True, random_state=42, return_X_y=False,
    ))
    records = [(i, text, int(label)) for i, (text, label)
               in enumerate(zip(dataset.data, dataset.target))]
    return records, list(dataset.target_names)


def prepare_data(records, excluded_texts=None):
    excluded_texts = set() if excluded_texts is None else excluded_texts
    nonempty = [row for row in records if row[1].strip()]
    labels_by_text = {}
    for _, text, label in nonempty:
        labels_by_text.setdefault(text, set()).add(label)
    conflicts = {text for text, labels in labels_by_text.items() if len(labels) > 1}

    seen = set()
    cleaned = []
    duplicate_rows = overlap_rows = conflict_rows = 0
    for row in nonempty:
        _, text, _ = row
        if text in excluded_texts:
            overlap_rows += 1
        elif text in conflicts:
            conflict_rows += 1
        elif text in seen:
            duplicate_rows += 1
        else:
            seen.add(text)
            cleaned.append(row)
    lengths = [len(row[1].split()) for row in nonempty]
    stats = {
        "raw": len(records), "empty": len(records) - len(nonempty),
        "duplicates_removed": duplicate_rows, "conflicting_texts": len(conflicts),
        "conflicting_rows_removed": conflict_rows, "train_overlap_removed": overlap_rows,
        "remaining": len(cleaned), "raw_class_counts": dict(Counter(r[2] for r in records)),
        "clean_class_counts": dict(Counter(r[2] for r in cleaned)),
        "word_lengths_nonempty": {
            "min": min(lengths, default=0), "median": median(lengths) if lengths else 0,
            "max": max(lengths, default=0),
        },
    }
    return cleaned, stats
