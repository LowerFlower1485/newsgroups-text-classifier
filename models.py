"""Заранее заданные эксперименты: меняем одну группу настроек за раз."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

EXPERIMENTS = [
    {"name": "baseline", "kind": "word", "C": 1.0},
    {"name": "regularization_0.1", "kind": "word", "C": 0.1},
    {"name": "regularization_10", "kind": "word", "C": 10.0},
    {"name": "word_bigrams", "kind": "bigram", "C": 1.0},
    {"name": "char_3_5", "kind": "char", "C": 1.0},
    {"name": "complement_nb", "kind": "word", "algorithm": "nb", "alpha": 1.0},
]


def build_model(config):
    if config["kind"] == "char":
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    elif config["kind"] == "bigram":
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    else:
        vectorizer = TfidfVectorizer()
    # Создаём классификатор; обученные веса появятся после fit().
    if config.get("algorithm") == "nb":
        classifier = ComplementNB(alpha=config["alpha"])
    else:
        classifier = LogisticRegression(C=config["C"], max_iter=1000)
    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])
