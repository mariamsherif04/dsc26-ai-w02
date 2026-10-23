"""Data audit for Rawaj Pay transfer extracts.

You write this file. Keep the signature exactly as it is: the grader calls it by name.
Allowed dependencies: numpy, pandas, scikit-learn (and the standard library).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
 
_MIN_LABELLED = 20
_MIN_VALUES = 20
_AUC_HI = 0.98
_AUC_LO = 0.02
_EPS = 1e-12

def _missing_mask(s: pd.Series) -> pd.Series:
    """True where a value is NaN/None/NaT/NA or a string that is empty or only whitespace."""
    blank = s.map(lambda v: isinstance(v, str) and v.strip() == "").astype(bool)
    return s.isna() | blank

def _is_extreme(auc: float) -> bool:
    return auc >= _AUC_HI - _EPS or auc <= _AUC_LO + _EPS
 
def _numeric_values(s: pd.Series, miss: pd.Series) -> pd.Series | None:
    """Non-missing values as floats if the column is numeric or boolean, else None."""
    kept = s[~miss]
    if pd.api.types.is_bool_dtype(s) or (
        pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_complex_dtype(s)
    ):
        return pd.to_numeric(kept).astype(float)
    if s.dtype == object and len(kept) and all(isinstance(v, (bool, np.bool_)) for v in kept):
        return kept.astype(float)  # booleans with None live in an object column
    return None

def _parse_times(s: pd.Series, miss: pd.Series) -> pd.Series:
    """Parse the non-missing values of a time column; unparseable ones become NaT."""
    kept = s[~miss]
    if len(kept) == 0:
        return pd.Series([], dtype="datetime64[ns, UTC]")
    kept = kept.map(lambda v: v.strip() if isinstance(v, str) else v)
    if kept.map(lambda v: isinstance(v, str)).all():
        return pd.to_datetime(kept, format="ISO8601", errors="coerce", utc=True)
    return pd.to_datetime(kept, errors="coerce", utc=True)


def audit(df: pd.DataFrame, *, target: str, id_col: str, time_col: str) -> dict:
    """Audit a transfer extract before anyone trains a model on it.
 
    Raises KeyError if `target`, `id_col` or `time_col` is not a column of `df`.
 
    Returns a dict with exactly these keys:
 
    - "n_rows" (int): number of rows.
    - "missing" ({column: float}): fraction missing, for every column. Missing means
      NaN/None, or a string that is empty or only whitespace.
    - "duplicate_ids" (list): sorted non-missing `id_col` values that appear more than once.
    - "duplicate_rows" (int): rows that exactly repeat an earlier row.
    - "constant_columns" (list): sorted columns other than `target` with at most one
      distinct non-missing value.
    - "unlabelled_rows" (int): rows where `target` is missing.
    - "positive_rate" (float): share of labelled rows with target == 1; 0.0 if none is labelled.
    - "bad_timestamps" (int): non-missing values in `time_col` that do not parse as dates.
    - "time_range" (list): [min, max] of the parsed timestamps as ISO strings, or [None, None].
    - "leakage_suspects" (list): sorted columns (not target, id_col or time_col) that look
      like leaks. See README.md for the exact rule.
    """
    for col in (target, id_col, time_col):
        if col not in df.columns:
            raise KeyError(col)
 
    n = len(df)
    masks = {c: _missing_mask(df[c]) for c in df.columns}
 
    missing = {c: (float(m.sum()) / n if n else 0.0) for c, m in masks.items()}
 
    ids = df[id_col][~masks[id_col]]
    counts = ids.value_counts()
    dup_ids = counts[counts > 1].index.tolist()
    try:
        dup_ids = sorted(dup_ids)
    except TypeError:
        dup_ids = sorted(dup_ids, key=str)
 
    duplicate_rows = int(df.duplicated().sum())
 
    constant = []
    for c in df.columns:
        if c == target:
            continue
        if df[c][~masks[c]].nunique(dropna=True) <= 1:
            constant.append(c)
    constant = sorted(constant)
 
    # --- labels ---
    labelled = ~masks[target]
    unlabelled_rows = int((~labelled).sum())
    y_all = pd.to_numeric(df[target][labelled], errors="coerce")
    n_lab = int(labelled.sum())
    positive_rate = float((y_all == 1).sum()) / n_lab if n_lab else 0.0
 
    # --- timestamps ---
    parsed = _parse_times(df[time_col], masks[time_col])
    n_present = int((~masks[time_col]).sum())
    good = parsed.dropna()
    bad_timestamps = n_present - len(good)
    if len(good):
        time_range = [good.min().isoformat(), good.max().isoformat()]
    else:
        time_range = [None, None]
 
    # --- leakage ---
    suspects: list = []
    if n_lab >= _MIN_LABELLED:
        y = (y_all == 1).astype(int)
        if y.nunique() == 2:
            lab_idx = y.index
            for c in df.columns:
                if c in (target, id_col, time_col):
                    continue
                col, m = df[c], masks[c]
                if col.map(lambda v: "<NA>" if pd.isna(v) or (isinstance(v, str) and not v.strip()) else v)\
                        .nunique(dropna=False) <= 1:
                    continue  # same in every row (missing counts as a value): never a suspect
                flagged = False
 
                # (a) value leak
                vals = _numeric_values(col.loc[lab_idx], m.loc[lab_idx])
                if vals is not None and len(vals) >= _MIN_VALUES:
                    yy = y.loc[vals.index]
                    if yy.nunique() == 2:
                        flagged = _is_extreme(roc_auc_score(yy, vals))
 
                # (b) missingness leak
                if not flagged:
                    ind = m.loc[lab_idx].astype(int)
                    flagged = _is_extreme(roc_auc_score(y, ind))
 
                if flagged:
                    suspects.append(c)
    suspects = sorted(suspects)
 
    return {
        "n_rows": int(n),
        "missing": missing,
        "duplicate_ids": dup_ids,
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant,
        "unlabelled_rows": unlabelled_rows,
        "positive_rate": positive_rate,
        "bad_timestamps": int(bad_timestamps),
        "time_range": time_range,
        "leakage_suspects": suspects,
    }
 