import math

import numpy as np
import pandas as pd
import pytest

from rawaj.audit import audit
from rawaj.metrics import (best_threshold, confusion_at, expected_cost,
                           precision_at_k, precision_recall_at,
                           threshold_for_precision)


def frame(n=100, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    return pd.DataFrame({
        "id": [f"t{i}" for i in range(n)],
        "t": ["2026-09-01 10:00"] * n,
        "y": y,
        "noise": rng.normal(size=n),
    })


def run(df):
    return audit(df, target="y", id_col="id", time_col="t")


# ---------- audit ----------
def test_missing_counts_blank_whitespace_none_nan():
    df = pd.DataFrame({"id": list("abcde"), "t": ["2026-09-01 10:00"] * 5, "y": [0, 1, 0, 1, 0],
                       "s": ["", "  ", "\t\n", None, "ok"],
                       "x": [1.0, np.nan, 3.0, 4.0, 5.0]})
    r = run(df)
    assert r["missing"]["s"] == pytest.approx(4 / 5)
    assert r["missing"]["x"] == pytest.approx(1 / 5)


def test_missing_key_columns_raise_keyerror():
    df = frame()
    for kw in ({"target": "nope", "id_col": "id", "time_col": "t"},
               {"target": "y", "id_col": "nope", "time_col": "t"},
               {"target": "y", "id_col": "id", "time_col": "nope"}):
        with pytest.raises(KeyError):
            audit(df, **kw)


def test_constant_columns_ignore_missing_and_target():
    df = frame()
    df["c"] = ["EGP"] * 50 + [None] * 50
    df["blank"] = ""
    assert {"c", "blank"} <= set(run(df)["constant_columns"])
    df["y"] = 1
    assert "y" not in run(df)["constant_columns"]


def test_unlabelled_and_positive_rate():
    df = frame(10)
    df.loc[:3, "y"] = None
    r = run(df)
    assert r["unlabelled_rows"] == 4
    assert r["positive_rate"] == pytest.approx(3 / 6)
    df["y"] = None
    assert run(df)["positive_rate"] == 0.0


def test_bad_timestamps_and_range():
    df = frame(6)
    df.loc[0, "t"] = "2026-02-30 10:00"
    df.loc[1, "t"] = "garbage"
    df.loc[2, "t"] = " "
    r = run(df)
    assert r["bad_timestamps"] == 2
    assert pd.Timestamp(r["time_range"][0]) == pd.Timestamp("2026-09-01 10:00", tz="UTC")
    df["t"] = "nope"
    r = run(df)
    assert r["bad_timestamps"] == 6 and r["time_range"] == [None, None]


def test_duplicates_ignore_missing_ids():
    df = frame(6)
    df.loc[1, "id"] = df.loc[0, "id"]
    df.loc[2:3, "id"] = None
    assert run(df)["duplicate_ids"] == ["t0"]


def test_value_leaks_and_inverse_and_bool():
    df = frame()
    df["copy"] = df["y"]
    df["inv"] = 1 - df["y"]
    df["flag"] = df["y"].astype(bool)
    assert {"copy", "inv", "flag"} <= set(run(df)["leakage_suspects"])
    assert "noise" not in run(df)["leakage_suspects"]


def test_missingness_leak_blank_only_for_positives_or_filled_only_for_positives():
    df = frame()
    df["filled_pos"] = np.where(df["y"] == 1, "2026-09-02", None)
    df["blank_pos"] = np.where(df["y"] == 1, "  ", "a")
    assert {"filled_pos", "blank_pos"} <= set(run(df)["leakage_suspects"])


def test_single_value_columns_are_never_suspects():
    df = frame()
    df["const"] = "EGP"
    df["allblank"] = ""
    assert not {"const", "allblank"} & set(run(df)["leakage_suspects"])


def test_leak_minimums():
    df = frame(100)
    df["copy"] = df["y"]
    assert run(df.head(19))["leakage_suspects"] == []        # < 20 labelled rows
    one = df.copy()
    one["y"] = 1
    assert run(one)["leakage_suspects"] == []                 # one class only
    few = df.copy()
    few["copy"] = np.where(np.arange(100) < 19, df["copy"], np.nan)
    run(few)  # fewer than 20 non-missing values: check (a) is skipped and must not crash


# ---------- metrics ----------
def test_precision_at_k_ties_stable_and_k_gt_n():
    assert precision_at_k([1, 0, 1, 0], [.5, .5, .5, .5], 1) == 1.0   # first in input order
    assert precision_at_k([0, 1, 0, 1], [.5, .5, .5, .5], 1) == 0.0
    assert precision_at_k([1, 0], [.9, .1], 99) == 0.5


def test_precision_recall_zero_division():
    assert precision_recall_at([0, 0], [.1, .2], 0.0) == (0.0, 0.0)   # no positives
    assert precision_recall_at([1, 1], [.1, .2], 0.0) == (1.0, 1.0)


def test_best_threshold_tie_prefers_highest_and_inf():
    assert best_threshold([0, 0], [.3, .6], cost_fp=1, cost_fn=1) == math.inf
    # equal cost at 0.2 and 0.9 -> keep the higher one
    assert best_threshold([1, 0], [.9, .2], cost_fp=1, cost_fn=1) == 0.9


def test_threshold_for_precision_non_monotonic_and_none():
    y = [1, 0, 1, 1]
    s = [.9, .8, .7, .6]
    assert threshold_for_precision(y, s, 0.75) == pytest.approx(0.6)
    assert threshold_for_precision([0, 0], [.1, .2], 0.5) is None


def test_expected_cost():
    assert expected_cost([1, 0, 1], [.9, .8, .1], 0.5, cost_fp=3, cost_fn=10) == 13


FUNCS = [
    lambda y, s: confusion_at(y, s, 0.5),
    lambda y, s: precision_recall_at(y, s, 0.5),
    lambda y, s: precision_at_k(y, s, 1),
    lambda y, s: expected_cost(y, s, 0.5, cost_fp=1, cost_fn=1),
    lambda y, s: best_threshold(y, s, cost_fp=1, cost_fn=1),
    lambda y, s: threshold_for_precision(y, s, 0.5),
]


@pytest.mark.parametrize("f", FUNCS)
@pytest.mark.parametrize("y,s", [([], []), ([1], [.1, .2]), ([2], [.1]), ([1], [float("nan")]),
                                 ([-1], [.1])])
def test_value_errors(f, y, s):
    with pytest.raises(ValueError):
        f(y, s)


def test_k_must_be_positive():
    for k in (0, -1):
        with pytest.raises(ValueError):
            precision_at_k([1], [.5], k)