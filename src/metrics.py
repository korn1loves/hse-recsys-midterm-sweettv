"""MAP@K as on Kaggle (ml_metrics.mapk semantics)."""
import numpy as np


def apk(actual, predicted, k=5):
    """AP@k = sum_i P(i) * rel(i) / min(|actual|, k); duplicate predictions are not re-credited."""
    if not actual:
        return 0.0
    actual = set(actual)
    predicted = list(predicted)[:k]
    hits, score = 0, 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            hits += 1
            score += hits / (i + 1.0)
    return score / min(len(actual), k)


def mapk(actual: dict, predicted: dict, k=5):
    """Mean AP@k over the users in `actual` (missing prediction -> AP 0)."""
    return float(np.mean([apk(a, predicted.get(u, []), k) for u, a in actual.items()]))
