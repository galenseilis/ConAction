'''
BSD 3-Clause License

Copyright (c) 2021, Galen Seilis
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

# TODO: Consider suggestions at https://stackoverflow.com/questions/35673895/type-hinting-annotation-pep-484-for-numpy-ndarray
# TODO: Ensure code follows style guide: https://numpydoc.readthedocs.io/en/latest/format.html
import numpy as np
from scipy.stats import rankdata

def grade_entropy(X):
    P = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            if np.all(X[i] < X[j]):
                P[i, j] += 1
    bins, counts = np.unique(np.sum(P, axis=0), return_counts=True)
    probs = counts / np.sum(counts)
    entropy = np.sum(probs * np.log(probs)) / (np.log(1/np.sum(counts)))
    return entropy

# TODO: Add Bessel correction
def minkowski_deviation(x: np.ndarray, order=2) -> float:
    '''
    Calculates the Minkowski deviation of
    order p. When the order = 2, it is the
    same as the standard deviation.
    '''
    result = x - np.mean(x)
    result = np.abs(x)
    result = np.power(result, order)
    result = np.mean(result)
    result = np.power(result, 1 / order)
    return result

def reflective_correlation(X: np.ndarray) -> float:
    '''
    Calculates the n-ary reflective correlation coefficient.
    When given an m x 2 data matrix, it is equivalent to the
    reflective correlation coefficient.

    Parameters
    ----------
        X : array_like
            The m x n data matrix.

    Returns
    -------
    r : float
        The calculated reflective correlation coefficient.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Reflective_correlation_coefficient

    Examples
    ----------
    '''
    numerator = np.prod(X, axis=1)
    numerator = np.sum(numerator)
    denominator = np.abs(X)
    denominator = np.power(denominator, X.shape[1])
    denominator = np.sum(denominator, axis=0)
    denominator = np.prod(denominator)
    denominator = np.power(denominator, 1 / X.shape[1])
    r = numerator / denominator
    return r

def pearson_correlation(X: np.ndarray) -> float:
    '''
    This function calculates the n-ary Pearson's r correlation
    coefficient. When given an m x 2 data matrix, it is equivalent
    to the Pearson's r correlation coefficient.

    Parameters
    ----------
        X : array_like
            The m x n data matrix.

    Returns
    -------
    r : float
        The calculated Pearson r correlation coefficient.

    References
    ----------
       https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

    Examples
    ----------
    '''
    transform = X - np.mean(X, axis=0)
    r = reflective_correlation(transform)
    return r

def circular_correlation(X: np.ndarray) -> float:
    '''
    This function calculates the n-ary circular correlation
    coefficient. When given an m x 2 data matrix, it is equivalent
    to the circular correlation coefficient.
    
    Parameters
    ----------
        X (array-like): m x n data matrix

    Returns
    -------
        r : float
            Circular correlation coefficient.

    References
    ----------
    .. [1] "Circular Correlation Coefficient", https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Circular_correlation_coefficient

    Examples
    ----------
    '''
    transform = X - np.mean(X, axis=0)
    transform = np.sin(transform)
    r = reflective_correlation(transform)
    return r

##def spearman_correlation(X: np.ndarray, method='average') -> float:
##    '''
##    This function calculates the n-ary Spearman correlation
##    coefficient. When given an m x 2 data matrix, it is equivalent
##    to the Spearman's Rho correlation coefficient.
##
##    The available data ranking options are directly from `scipy.stats.rankdata`.
##    
##    Parameters
##    ----------
##        X : array_like
##            m x n data matrix.
##        method : {'average', 'min', 'max', 'dense', 'ordinal'}, optional
##            The method used to assign ranks to tied elements.
##            The following methods are available (default is 'average'):
##          * 'average': The average of the ranks that would have been assigned to
##            all the tied values is assigned to each value.
##          * 'min': The minimum of the ranks that would have been assigned to all
##            the tied values is assigned to each value.  (This is also
##            referred to as "competition" ranking.)
##          * 'max': The maximum of the ranks that would have been assigned to all
##            the tied values is assigned to each value.
##          * 'dense': Like 'min', but the rank of the next highest element is
##            assigned the rank immediately after those assigned to the tied
##            elements.
##          * 'ordinal': All values are given a distinct rank, corresponding to
##            the order that the values occur in `a`.
##
##    Returns
##    -------
##        : float
##            Spearman's correlation coefficient.
##            
##    References
##    ----------
##    .. [1] "Spearman%27s_rank_correlation_coefficient", https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
##    .. [2] "scipy.stats.rankdata", https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html
##
##    '''
##    transform = rankdata(X, axis=0, method=method)
##    transform = transform - np.mean(transform, axis=0)
##    return reflective_correlation(transform)

# TODO: Implement
def angular_disimilarity(X: np.ndarray) -> float:
    '''
    Computes the n-ary angular disimilarity.
    When given an m x 2 data matrix, it is equivalent
    to the angular distance.

    References
    ----------
    .. [1] "Angular distance", https://en.wikipedia.org/wiki/Cosine_similarity#Angular_distance_and_similarity
    .. [2] "What is the exact and precise definition of an ANGLE?", https://math.stackexchange.com/questions/583066/what-is-the-exact-and-precise-definition-of-an-angle
    '''

# TODO: Implement 
def correlation_ratio(X: np.ndarray, y: np.ndarray) -> float:
    '''
    This function calculates the n-ary correlation ratio of a
    collection of response variables given their classes. The
    classic Fisher's correlation ratio is a special case.

    Parameters
    ----------
        X: array_like
            m x n data matrix
        y: array_like
            m-dimensional vector of class labels

    References
    ----------
    .. [1] "Correlation Ratio", https://en.wikipedia.org/wiki/Correlation_ratio

    Examples
    ----------
    '''
    global_means = np.mean(X, axis=0)
    classes = set(y)
    partitions_X = [X[y == c] for c in classes]

def misiak_correlation(x: np.ndarray, y: np.ndarray, X: np.ndarray) -> float:
    '''
    Misiak's n-inner correlation coefficient.

    Parameters
    ----------
        x: array_like
            1-D data vector
        y: array_like
            1-D data vector
        X: array_like
            m x n data matrix
    '''
    G = np.empty((X.shape[1]+1, X.shape[1]+1))
    G[0, 0] = x @ y
    G[0, 1:] = (x.reshape(-1,1).T @ X).flatten()
    G[1:, 0] = (y.reshape(-1,1).T @ X).flatten()
    G[1:, 1:] = X.T @ X
    numerator = np.linalg.det(G)
    G[0, 0] = x @ x
    G[1:, 0] = (x.reshape(-1,1).T @ X).flatten()
    denominator = np.linalg.det(G)
    G[0, 0] = y @ y
    G[0, 1:] = (y.reshape(-1,1).T @ X).flatten()
    G[1:, 0] = (y.reshape(-1,1).T @ X).flatten()
    denominator = denominator * np.linalg.det(G)
    denominator = np.sqrt(denominator)
    result = numerator / denominator
    return result

def multisemideviation(X: np.ndarray, p=1) -> float:
    '''
    This function calculates the multisemideviation
    which is the multisemimetric between a collection
    of random variables from their expectations. The
    multisemimetric is induced by the multiseminorm,
    which is a generalization of the notion of a
    seminorm.

    Parameters
    ----------
        X: array_like
            m x n data matrix
    Returns
    ----------
        : float
            Multisemideviation score.

    References
    ----------
    .. [1] "Seminorm", https://mathworld.wolfram.com/Seminorm.html
    .. [2] "Seminorm", https://en.wikipedia.org/wiki/Seminorm
    '''
    result = X - np.mean(X, axis=0)
    result = np.prod(result, axis=1)
    result = np.power(result, p)
    result = np.mean(result)
    result = np.power(result, 1/p)
    return result

# TODO: Implement math
def taylor_correlation(X: np.ndarray) -> float:
    '''
    References
    ----------
    .. [1] https://arxiv.org/abs/2003.02561
    '''
    d = X.shape[1]
    result = np.corrcoef(X)
    result = np.linalg.eigvals(result)
    result = np.std(result)
    result = result / np.sqrt(d)
    return result

def trencevski_melceski_correlation(X: np.ndarray, Y: np.ndarray) -> np.float64:
    '''
    Generalized n-inner product correlation coefficient.

    Computes a correlation coefficient based
    on Trencevski and Melceski 2006.

    Parameters
    ----------
        X: array_like
            m x n data matrix
        Y: array_like
            m x n data matrix
    Returns
    ----------
        : float
            Correlation score.
    '''
    numerator = np.linalg.det(X.T @ Y)
    denominator = np.linalg.det(X.T @ X) *\
                  np.linalg.det(Y.T @ Y)
    denominator = np.sqrt(denominator)
    result = numerator / denominator
    return result

def wang_zheng_correlation(X: np.ndarray) -> float:
    '''
    References
    ----------
    .. [1] ArXiv:1401.4827v6
    '''
    result = X.T
    result = np.corrcoef(result)
    result = np.linalg.det(result)
    result = 1 - result
    return result

if __name__ == '__main__':
    x = np.random.normal(size=100)
    y = np.random.normal(size=100)
    X = np.random.normal(size=100*3).reshape(100,3)
    
