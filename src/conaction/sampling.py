'''
The Sampling module provides functions for resampling and search methods
suitable for functions found in the Estimators module.

References
----------
.. [1] "Resampling.", https://en.wikipedia.org/wiki/Resampling_(statistics)
'''

import numpy as np
import tqdm

def permute_columns(x):
    '''
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
    '''
    ix_i = np.random.sample(x.shape).argsort(axis=0)
    ix_j = np.tile(np.arange(x.shape[1]), (x.shape[0], 1))
    return x[ix_i, ix_j]

def statistic_permute(X, stat_func=lambda x: x, iters=100):
    '''
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
    '''
    y = []
    for r in tqdm.tqdm(range(iters), total=iters):
        y.append(stat_func(permute_columns(X)))
    return y

def powerset_search():
    '''
    Computes a set function on the whole powerset
    of sets of variables.

    Raises
    ------
    NotImplementedError
    '''
    raise NotImplementedError

def isoarity_search():
    '''
    For a given arity of relation,
    compute set function on all sets
    whose cardinality equals that arity.

    Raises
    ------
    NotImplementedError
    '''
    raise NotImplementedError

def heuristic_monotonic_closure_search():
    '''
    Compute a search of statistical hypotheses
    "as-if" statistical significiance had a
    downward closure property on the set of
    variables.

    Raises
    ------
    NotImplementedError
    '''
    raise NotImplementedError
