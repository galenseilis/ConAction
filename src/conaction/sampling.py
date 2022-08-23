"""
The Sampling module provides functions for resampling and search methods
suitable for functions found in the Estimators module.

References
----------
.. "Resampling.", https://en.wikipedia.org/wiki/Resampling_(statistics)
"""

from itertools import combinations

import numpy as np
import tqdm


def permute_columns(x):
    """
    Permutes the columns of a data matrix.

    Parameters
    ----------
    x : array-like
        An m x n data matrix.

    Returns
    -------
    : array-like
        Permuted matrix.

    References
    ----------
    .. [1] "Permutation.", https://en.wikipedia.org/wiki/Permutation
    .. [2] "Random permutation.", https://en.wikipedia.org/wiki/Random_permutation

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> data = np.arange(100).reshape(10,10)
    >>> permute_columns(data)
    array([[60, 61, 82, 43, 34, 75, 16, 97, 78, 99],
       [30, 81, 92, 53, 14, 15, 26, 87, 48, 69],
       [80, 41, 32, 63, 24, 55, 46, 67, 58, 79],
       [90, 51, 22,  3, 64, 95, 76, 77, 28, 59],
       [40, 71, 12, 33, 54, 85,  6, 47, 88, 49],
       [ 0, 11,  2, 73, 94, 65, 86, 57, 18,  9],
       [50, 91, 62, 83,  4, 35, 96, 37, 98, 29],
       [10,  1, 42, 93, 84, 25, 36, 17, 68, 39],
       [70, 31, 72, 23, 44,  5, 56,  7, 38, 19],
       [20, 21, 52, 13, 74, 45, 66, 27,  8, 89]])
    """
    ix_i = np.random.sample(x.shape).argsort(axis=0)
    ix_j = np.tile(np.arange(x.shape[1]), (x.shape[0], 1))
    return x[ix_i, ix_j]


def statistic_permute(X, stat_func=lambda x: x, iters=100):
    """
    Performs a permutation Monte Carlo of a statistical
    function applied to a given dataset.

    Parameters
    ----------
    X : array-like
        An m x n data matrix.
    stat_func : function
        Function to be calculated on new samples. (default=lambda x: x)

    Returns
    -------
    y : list
        Resulting permutation statistics.

    Notes
    -----
    This function wraps around the permute_columns function to
    sample a function of a data matrix multiple times under
    pseudo-randomly sampled permutations within the columns.

    References
    ----------
    .. [1] "Resampling.", https://en.wikipedia.org/wiki/Resampling_(statistics)
    .. [2] "Permutation testing.", https://en.wikipedia.org/wiki/Permutation_test
    .. [3] "Random permutation.", https://en.wikipedia.org/wiki/Random_permutation

    Examples
    --------
    >>> import numpy as np
    >>> data = np.arange(9).reshape(3,3)
    >>> statistic_permute(data, np.mean, 2)
    [4.0, 4.0]
    """
    y = []
    for r in tqdm.tqdm(range(iters), total=iters):
        y.append(stat_func(permute_columns(X)))
    return y


def powerset_search(f, X, leastarity=1):
    """
    Computes a set function on the whole powerset
    of sets of variables.

    Parameters
    ----------
    f : function
        Estimator
    X : np.ndarray-like
        Data matrix

    Returns
    -------
    results : dict
        Computing values of f.
    """
    n = X.shape[1]
    for arity in range(leastarity, n + 1):
        for comb in combinations(range(n), arity):
            results[comb] = f(X[:, comb])
    return results


def isoarity_search(f, X, arity):
    """
    For a given arity of relation,
    compute set function on all sets
    whose cardinality equals that arity.

    Parameters
    ----------
    f : function
        Estimator
    X : np.ndarray-like
        Data matrix
    arity : int
        Arity to compute f at; number of variables.

    Returns
    -------
    results : dict
        Results of computing function f on columns of X.
    """
    n = X.shape[1]
    results = {}
    for comb in combinations(range(n), arity):
        results[comb] = f(X[:, comb])
    return results


def heuristic_downward_closure_search(f, X, tau=0.1, abscomp=False, leastarity=2):
    """
    Search via the downward closure heuristic.

    Parameters
    ----------
    f : function
        A permutation-invariant multiary function.
    X : ndarray
        m x n data matrix.
    tau : float
        Downward closure threshold.
    abscomp : Bool
        Whether to compare threshold to absolute value of f.
    leastarity : int (default=2)
        The function arity to start search.

    Returns
    -------
    d : dict[dict]
        Computing values of f.
    """
    d = {leastarity: {}}
    for comb in combinations(range(X.shape[1]), leastarity):
        d[leastarity][frozenset(comb)] = f(X[:, comb])

    for arity in range(leastarity + 1, X.shape[1] + 1):
        print(arity)
        d[arity] = {}
        prev_keys = list(d[arity - 1].keys())
        for i, key_i in enumerate(tqdm(prev_keys)):
            if d[arity - 1][key_i] > tau or (
                abscomp and np.abs(d[arity - 1][key_i]) > tau
            ):
                for j in range(i + 1, len(prev_keys)):
                    key_j = prev_keys[j]
                    if d[arity - 1][key_j] > tau or (
                        abscomp and np.abs(d[arity - 1][key_i]) > tau
                    ):
                        union = key_i.union(key_j)
                        if len(union) == arity:
                            d[arity][union] = f(X[:, tuple(union)])
                        else:
                            continue
                    else:
                        continue
            else:
                continue
        if not d[arity]:
            del d[arity]
            return d
        else:
            print(f"Found {len(d[arity])} results.")
    return d
