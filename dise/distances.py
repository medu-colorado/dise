import numpy as np
from numpy.linalg import inv
from scipy.stats import wasserstein_distance
from sklearn.neighbors import NearestNeighbors


def percentile_difference(A, B=None):
    """
    Percentile difference of each column.

    Regression (B=None) case: |P75(A) - P25(A)| (IQR).
        Takes the inter quartile range (IQR), 75th percentile - 25th percentile.
    Classification case:      |P75(A) - P75(B)|.
        Takes the 75th Percentile of A and subtracts 75th Percentile of B.

    Parameters
    ----------
    A : ndarray, shape (n, p)
    B : ndarray, shape (m, p) or None

    Returns
    -------
    ndarray of shape (p,)
    """
    if B is None:
        return np.abs(np.percentile(A, 75, axis=0) - np.percentile(A, 25, axis=0))
    return np.abs(np.percentile(A, 75, axis=0) - np.percentile(B, 75, axis=0))


def wasserstein_vector(A, B=None):
    """
    Wasserstein distance (or Earth Mover's Distance) measures how much it
    takes to 'move' the mass of one probability distribution into another.

    Regression: Computes Wasserstein Distance between
        lower portion of the distribution (<=P25) and the
        upper portion of the distribution (>=P75).

    Classification: Computes the Wasserstein distance between the
        two class distributions.

    Parameters
    ----------
    A : ndarray, shape (n, p)
    B : ndarray, shape (m, p) or None

    Returns
    -------
    ndarray of shape (p,)
    """
    if B is None:
        X = np.asarray(A)
        results = []
        for i in range(X.shape[1]):
            col = X[:, i]
            x1 = col[col <= np.quantile(col, 0.25)]
            x2 = col[col >= np.quantile(col, 0.75)]
            results.append(wasserstein_distance(x1, x2))
        return np.array(results)
    return np.array([wasserstein_distance(A[:, i], B[:, i]) for i in range(A.shape[1])])


def mahalanobis_delta(A, B=None, type_map=None):
    """
    A Sensitivity vector based on covariance structure.

    Discrete columns are excluded from the covariance computation and
    receive 0.0 in the output vector.

    Regression: square root of precision matrix diagonal. This is the inverse of covariance
    per feature. Note - considering adjusting this.

    Classification: absolute Mahalanobis-weighted mean difference.

    Parameters
    ----------
    A : ndarray, shape (n, p)
    B : ndarray, shape (m, p) or None
    type_map : dict[int, str] or None

    Returns
    -------
    ndarray of shape (p,)
    """
    n_cols = A.shape[1]

    if type_map:
        cont_cols = np.array(
            [i for i in range(n_cols) if type_map.get(i) != "discrete"]
        )
    else:
        cont_cols = np.arange(n_cols)

    if len(cont_cols) == 0:
        return np.zeros(n_cols)

    A_cont = A[:, cont_cols].astype(float)

    if B is None:
        cov = np.atleast_2d(np.cov(A_cont.T))
        cov_inv = inv(cov + np.eye(cov.shape[0]) * 1e-6)
        # TODO try using the mean here for A.
        partial = np.sqrt(np.diag(cov_inv))
    else:
        B_cont = B[:, cont_cols].astype(float)
        mean_A = A_cont.mean(axis=0)
        mean_B = B_cont.mean(axis=0)
        cov = np.atleast_2d(np.cov(np.vstack([A_cont, B_cont]).T))
        cov_inv = inv(cov + np.eye(cov.shape[0]) * 1e-6)
        partial = np.abs((mean_A - mean_B) @ cov_inv)

    result = np.zeros(n_cols)
    result[cont_cols] = partial
    return result


def minimal_counterfactual(A, B=None, y=None):
    """
    Uses nearest-neighbour distances between samples as a distance metric.

    Regression: compares samples above and below the median output.

    Classification: compares the two class sub-matrices.

    Parameters
    ----------
    A : ndarray, shape (n, p)
    B : ndarray, shape (m, p) or None
    y : ndarray, shape (n,) — used only in the regression case

    Returns
    -------
    ndarray of shape (p,)
    """
    if B is None:
        higher = A[y > np.median(y)]
        lower = A[y <= np.median(y)]
        if len(higher) == 0 or len(lower) == 0:
            return np.std(A, axis=0)
        nbrs = NearestNeighbors(n_neighbors=1).fit(higher)
        _, indices = nbrs.kneighbors(lower)
        diffs = higher[indices[:, 0]] - lower
        return np.mean(np.abs(diffs), axis=0)

    nbrs = NearestNeighbors(n_neighbors=1).fit(B)
    _, indices = nbrs.kneighbors(A)
    diffs = B[indices[:, 0]] - A
    return np.mean(np.abs(diffs), axis=0)
