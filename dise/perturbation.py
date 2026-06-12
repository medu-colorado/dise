import numpy as np


def _discrete_alternatives(X, col_idx, current_val):
    """All observed values for a discrete column other than current_val."""
    all_vals = np.unique(X[:, col_idx])
    return all_vals[all_vals != current_val]


def _perturb_sample(
    sample_X, feature_group, dist_vals, type_map, direction, scale, X_ref
):
    """
    Apply a single directional perturbation to a sample.

    Continuous features are shifted by ±dist_vals[f] * scale.
    Discrete features are swapped to the next/previous observed category.

    Parameters
    ----------
    sample_X  : ndarray, shape (1, p)
    feature_group : array-like of int
    dist_vals : ndarray, shape (p,)
    type_map  : dict[int, str]
    direction : "plus" | "minus"
    scale     : float
    X_ref     : ndarray, shape (n, p) — used to look up discrete alternatives

    Returns
    -------
    perturbed_X : ndarray, shape (1, p)
    move_values : dict {col: (type, old_val, new_val)}
    """
    sample_X = sample_X.copy()
    move_values = {}

    for f in feature_group:
        if type_map.get(f) == "discrete":
            current_val = sample_X[0, f]
            alternatives = _discrete_alternatives(X_ref, f, current_val)
            if len(alternatives) == 0:
                move_values[f] = ("discrete_no_alternative", current_val, current_val)
                continue
            if direction == "plus":
                candidates = alternatives[alternatives > current_val]
                new_val = candidates.min() if len(candidates) else alternatives.max()
            else:
                candidates = alternatives[alternatives < current_val]
                new_val = candidates.max() if len(candidates) else alternatives.min()
            move_values[f] = ("discrete", current_val, new_val)
            sample_X[0, f] = new_val
        else:
            delta = dist_vals[f] * scale * (1 if direction == "plus" else -1)
            move_values[f] = (
                "continuous",
                float(sample_X[0, f]),
                float(sample_X[0, f] + delta),
            )
            sample_X[0, f] += delta

    return sample_X, move_values


def move_group_classification(
    model,
    sample_X,
    feature_group,
    dist_vals,
    type_map,
    first_class,
    sample_idx,
    direction,
    scale,
    X_ref,
    unpack_fn,
):
    """
    Perturb a sample and record whether the predicted class changed.

    Returns
    -------
    dict with perturbation metadata.
    """
    perturbed, move_values = _perturb_sample(
        sample_X, feature_group, dist_vals, type_map, direction, scale, X_ref
    )
    pred_after = unpack_fn(model.predict(perturbed))
    return {
        "sample_idx": sample_idx,
        "feature_group": list(feature_group),
        "move_values": move_values,
        "original_class": first_class,
        "new_class": pred_after,
        "changed": pred_after != first_class,
        "direction": direction,
    }


def move_group_regression(
    model,
    sample_X,
    feature_group,
    dist_vals,
    type_map,
    sample_idx,
    direction,
    scale,
    X_ref,
    y_std,
    unpack_fn,
):
    """
    Perturb a sample and record the change in predicted output.

    Returns
    -------
    dict with perturbation metadata.
    """
    pred_before = unpack_fn(model.predict(sample_X))
    perturbed, move_values = _perturb_sample(
        sample_X, feature_group, dist_vals, type_map, direction, scale, X_ref
    )
    pred_after = unpack_fn(model.predict(perturbed))
    change = pred_after - pred_before
    return {
        "sample_idx": sample_idx,
        "feature_group": list(feature_group),
        "move_values": move_values,
        "original_y": float(pred_before),
        "new_y": float(pred_after),
        "change": float(change),
        "threshold_met_increase": change >= y_std,
        "threshold_met_decrease": change <= -y_std,
        "direction": direction,
    }
