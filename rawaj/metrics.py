"""Decision metrics for a mule-risk score.

You write this file. Keep the signatures exactly as they are: the grader calls them by name.

Rules for every function:
- y_true and scores are lists or 1-D arrays. Raise ValueError if the lengths differ, the
  input is empty, a label is not 0 or 1, or a score is NaN.
- A case is flagged when score >= threshold.
- Precision is 0.0 when nothing is flagged; recall is 0.0 when there are no positives.
"""
from __future__ import annotations

import math

import numpy as np


def _validate(y_true, scores) -> tuple[np.ndarray, np.ndarray]:
    """Return (y, s) as a bool array of labels and a float array of scores, or raise ValueError."""
    try:
        y = np.asarray(y_true)
        s = np.asarray(scores)
    except Exception as exc:  # ragged input and the like
        raise ValueError(f"could not read inputs: {exc}") from exc
    if y.ndim != 1 or s.ndim != 1:
        raise ValueError("y_true and scores must be 1-D")
    if len(y) != len(s):
        raise ValueError("y_true and scores must have the same length")
    if len(y) == 0:
        raise ValueError("inputs must not be empty")
    if y.dtype.kind not in "biuf" or s.dtype.kind not in "biuf":
        raise ValueError("y_true and scores must be numeric")
    yf = y.astype(float)
    sf = s.astype(float)
    if np.isnan(sf).any():
        raise ValueError("scores must not contain NaN")
    if not np.all((yf == 0) | (yf == 1)):  # NaN labels fail this test too
        raise ValueError("labels must be 0 or 1")
    return yf == 1, sf


def _counts(y: np.ndarray, s: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
    flagged = s >= threshold
    tp = int(np.sum(flagged & y))
    fp = int(np.sum(flagged & ~y))
    fn = int(np.sum(~flagged & y))
    tn = int(np.sum(~flagged & ~y))
    return tp, fp, tn, fn


def confusion_at(y_true, scores, threshold: float) -> dict:
    """Return {"tp": int, "fp": int, "tn": int, "fn": int} at `threshold`."""
    y, s = _validate(y_true, scores)
    tp, fp, tn, fn = _counts(y, s, threshold)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def precision_recall_at(y_true, scores, threshold: float) -> tuple[float, float]:
    """Return (precision, recall) at `threshold`."""
    y, s = _validate(y_true, scores)
    tp, fp, _, fn = _counts(y, s, threshold)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(precision), float(recall)


def precision_at_k(y_true, scores, k: int) -> float:
    """Share of positives among the k highest scores.

    ValueError if k <= 0; if k > n, use n. Rank by score descending; ties keep their
    original order.
    """
    y, s = _validate(y_true, scores)
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        if isinstance(k, (float, np.floating)) and float(k).is_integer():
            k = int(k)
        else:
            raise ValueError("k must be an integer")
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(int(k), len(y))
    order = np.argsort(-s, kind="stable")  # descending; stable keeps input order on ties
    return float(np.sum(y[order[:k]]) / k)


def expected_cost(y_true, scores, threshold: float, *, cost_fp: float, cost_fn: float) -> float:
    """fp * cost_fp + fn * cost_fn at `threshold`."""
    y, s = _validate(y_true, scores)
    _, fp, _, fn = _counts(y, s, threshold)
    return float(fp * cost_fp + fn * cost_fn)


def best_threshold(y_true, scores, *, cost_fp: float, cost_fn: float) -> float:
    """Threshold with the lowest expected cost.

    Candidates are every distinct score plus math.inf ("flag nothing"). On a tie in cost,
    return the highest threshold (fewer flags for the same cost).
    """
    y, s = _validate(y_true, scores)
    pos = np.sort(s[y])
    neg = np.sort(s[~y])
    # Candidates in DESCENDING order so argmin (first minimum) keeps the highest on a tie.
    cands = np.concatenate(([math.inf], np.unique(s)[::-1]))
    tp = len(pos) - np.searchsorted(pos, cands, side="left")   # positives with score >= t
    fp = len(neg) - np.searchsorted(neg, cands, side="left")   # negatives with score >= t
    fn = len(pos) - tp
    costs = fp * cost_fp + fn * cost_fn
    return float(cands[int(np.argmin(costs))])


def threshold_for_precision(y_true, scores, min_precision: float) -> float | None:
    """Lowest distinct score whose precision is >= min_precision, or None if none reaches it."""
    y, s = _validate(y_true, scores)
    pos = np.sort(s[y])
    neg = np.sort(s[~y])
    cands = np.unique(s)  # ascending: first hit is the lowest threshold
    tp = len(pos) - np.searchsorted(pos, cands, side="left")
    fp = len(neg) - np.searchsorted(neg, cands, side="left")
    for t, a, b in zip(cands, tp, fp):
        if a / (a + b) >= min_precision:  # a + b >= 1 because t is an observed score
            return float(t)
    return None