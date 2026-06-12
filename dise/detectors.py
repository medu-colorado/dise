import numpy as np
import pandas as pd


def column_check(X_i, count_threshold=20):
    """
    Determines whether a column is continuous or discrete.

    Parameters
    ----------
    X_i : array-like
        A single feature column (NumPy array, Pandas Series, or list).

    count_threshold: int
        The unique number of values needed to consider data continuous or discrete.

    Returns
    -------
    "discrete" or "continuous"
    """
    s = pd.Series(np.asarray(X_i).ravel())

    # Mixed-dtype arrays (e.g. from get_dummies + .values) come in as object —
    # attempt a numeric cast before any dtype checks
    if s.dtype == object:
        s = pd.to_numeric(s, errors="coerce")
        if s.isna().mean() > 0.5:
            return "discrete"

    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return "discrete"

    s = s.dropna()
    if s.empty:
        return "discrete"

    unique_count = s.nunique()

    if unique_count <= 2:
        return "discrete"

    # discrete if low cardinality, continuous if high
    is_integer_like = (
        pd.api.types.is_integer_dtype(s) or s.eq(s.astype(int, errors="ignore")).all()
    )
    if is_integer_like:
        return "continuous" if unique_count > count_threshold else "discrete"

    # Genuine float → continuous
    return "continuous"
