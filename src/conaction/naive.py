from collections import Counter
from copy import deepcopy
import functools
import math
from operator import mul
import statistics

import numpy as np

def kendall_tau(X, method='A'):
    '''
    Multilinear Kendall's tau

    Parameters
    ----------
    X : 2d list
        Data table (columns index variables, rows index samples)

    Returns
    -------
      : float
          Kendall's tau score
    '''
    concordant = 0
    discordant = 0

    for i, xi in enumerate(X):
        for j, xj in enumerate(X):
            if i < j:
                left_compare = [xik < xjk for xik, xjk in zip(xi, xj)]
                middle_compare = [xik == xjk for xik, xjk in zip(xi, xj)]
                right_compare = [xik > xjk for xik, xjk in zip(xi, xj)]
                if (all(left_compare) or all(right_compare)):
                    concordant += 1
                elif any(middle_compare):
                    continue
                else:
                    discordant += 1
            else:
                continue
    if method == 'A':
        return (concordant - discordant) / ((len(X) **2 - len(X)) / 2)
    elif method == 'B':
        raise NotImplementedError ('Method B is not implemented yet.')
    elif method == 'C':
        m = min((len(X), len(list(zip(*X)))))
        result =  2 * (concordant - discordant)
        result /= len(X)**2
        result /= (m-1) / m
        return result
    

def grade_entropy(X):
    '''
    Grade entropy of strict
    product order.

    Parameters
    ----------
    X : 2d list
        Data table (columns index variables, rows index samples)

    Returns
    -------
      : float
          Grade entropy score
    '''
    scores = []
    for i, xi in enumerate(X):
        score_i = 0
        for k, xk in enumerate(X):
            pairs = [xij < xkj for xij, xkj in zip(xi, xk)]
            if all(pairs):
                score_i += 1
        scores.append(score_i)
    counts = Counter(scores)
    total_score = sum(counts.values())
    probs = [score / total_score for score in counts.values()]
    surprises = [-p * math.log(p) for p in probs]
    entropy = sum(surprises)
    normalized_entropy = entropy / math.log(len(scores))
    return normalized_entropy
    

def reflective_correlation(x):
    '''
    Naive implementation of multilinear
    reflective correlation coefficient.

    Parameters
    ----------
    x : 2d list
        Data table (columns index variables, rows index samples)

    Returns
    -------
      : float
          correlation score
    '''
    order = len(x[0])
    xt = list(zip(*x))

    def mean(x):
        return sum(x) / len(x)

    abs_xt = map(lambda y: list(map(abs, y)), xt)
    SPR = map(lambda y: list(map(lambda z: pow(z,order), y)), abs_xt)
    MPE = map(mean, SPR)
    denom = pow(functools.reduce(mul, MPE, 1), 1/order)
    cov = list(map(lambda x: functools.reduce(mul, x, 1), zip(*xt)))
    cov = sum(cov) / len(cov)
    return  cov / denom

def pearson_correlation(x):
    '''
    Naive implementation of multilinear
    Pearson correlation coefficient.

    Parameters
    ----------
    x : 2d list
        Data table (columns index variables, rows index samples)

    Returns
    -------
      : float
          correlation score
    '''
    order = len(x[0])
    xt = list(zip(*x))

    def mean(x):
        return sum(x) / len(x)
    
    def residual(x):
        y, yhat = x
        return [yi - yhat for yi in y]

    means = map(lambda x: sum(x) / len(x), xt)
    residuals = list(map(residual, zip(xt, means)))
    abs_residuals = map(lambda y: list(map(abs, y)), residuals)
    SPR = map(lambda y: list(map(lambda z: pow(z,order), y)), abs_residuals)
    MPE = map(mean, SPR)
    denom = pow(functools.reduce(mul, MPE, 1), 1/order)
    cov = list(map(lambda x: functools.reduce(mul, x, 1), zip(*residuals)))
    cov = sum(cov) / len(cov)
    return  cov / denom

def circular_correlation(x):
    '''
    Naive implementation of multivariate
    circular correlation coefficient.

    Parameters
    ----------
    x : 2d list
        Data table (columns index variables, rows index samples)

    Returns
    -------
      : float
          correlation score
    '''
    xt = list(zip(*x))

    def mean(x):
        return sum(x) / len(x)
    
    def residual(x):
        y, yhat = x
        return [yi - yhat for yi in y]

    means = map(lambda x: sum(x) / len(x), xt)
    residuals = list(map(residual, zip(xt, means)))

    transformed = map(lambda y: list(map(math.sin, y)), residuals)
    y = list(zip(*transformed))
    return reflective_correlation(y)

def signum_correlation(X):
    '''
    Naive implementation of multivariate
    signum correlation coefficient.

    Parameters
    ----------
    x : 2d list
        Data table (columns index variables, rows index samples)

    Returns
    -------
      : float
          correlation score

    Examples
    --------
    >>> import numpy as np
    >>> X = np.arange(100).reshape(10,10)
    >>> X = [[float(j) for j in i] for i in X]
    >>> signum_correlation(X)
    1.0
    '''
    def sign(x):
        if x > 0:
            return 1.0
        elif x < 0:
            return -1.0
        else:
            return 0
        
    def transform(x):
        y, yhat = x
        return [sign(yi - yhat) for yi in y]

    Xt = list(zip(*X))
    means = [statistics.mean(xi) for xi in Xt]
    transformed = list(map(transform, zip(Xt, means)))
    y = list(zip(*transformed))
    return reflective_correlation(y)

def spearman_correlation(x):
    '''
    Naive implementation of multilinear
    Spearman correlation coefficient.

    Parameters
    ----------
    x : 2d list
        Data table (columns index variables, rows index samples)

    Returns
    -------
      : float
          correlation score
    '''
    xt = list(zip(*x))
    transformed = list(map(rankdata, xt))
    y = list(zip(*transformed))
    return pearson_correlation(y)

def minkowski_deviation(x, order=2):
    '''
    Naive implementation Minkowski
    deviation of order p.

    Parameters
    ----------
    x : 1d list
        Data table (columns index variables, rows index samples)
    p : int or float
        Order of deviation

    Returns
    -------
      : float
          Score
    '''
    mu = statistics.mean(x)
    pdev = [abs(xi - mu)**order for xi in x]
    return statistics.mean(pdev) ** (1 / order)

def misiak_correlation(x, y, X):
    '''
    Naive implementation of Misiak
    correlation.

    Parameters
    ----------
    x : 1d list
        Data vector
    y : 1d list
        Data vector
    X : 2d list
        Data table (columns index variables, rows index samples)

    Returns
    -------
      : float
          correlation score
    '''
    if len(set([len(i) for i in X])) != 1:
        raise ValueError('Incorrect shape.')
    m = len(X)+1
    n = len(X[0])+1
    Xt = list(zip(*X))
    M = matmult(Xt, X)
    G = [[None] * n] + [[None] + row for row in M]
    G[0][0] = dot(x,y)
    for i,xi in enumerate(Xt):
        G[0][i+1] = dot(x, xi)
        G[i+1][0] = dot(y,xi)
    numerator = determinant(G)
    G = [[None] * n] + [[None] + row for row in M]
    G[0][0] = dot(x,x)
    for i,xi in enumerate(Xt):
        G[0][i+1] = dot(x, xi)
        G[i+1][0] = dot(x,xi)
    denominator = determinant(G)
    G = [[None] * n] + [[None] + row for row in M]
    G[0][0] = dot(y, y)
    for i,xi in enumerate(Xt):
        G[0][i+1] = dot(y, xi)
        G[i+1][0] = dot(y, xi)
    denominator *= determinant(G)
    denominator = math.sqrt(denominator)
    return numerator / denominator

def multisemideviation(X, p=1):
    '''
    Naive implementation of multisemideviation
    of order p.

    Parameters
    ----------

    X : 2d list
        Data table (columns index variables, rows index samples)

    Returns
    -------
      : float
          Multisemideviation score
    '''
    Xt = list(zip(*X))
    means = [statistics.mean(xj) for xj in Xt]

    def residual(x):
        y, yhat = x
        return [abs(yi - yhat)**p for yi in y]

    pdevs = list(map(residual, zip(Xt, means)))
    pcov = list(map(lambda x: functools.reduce(mul, x, 1), zip(*pdevs)))
    pcov = statistics.mean(pcov)
    return pcov ** (1 / p)

def taylor_correlation(X):
    '''
    Naive implementation of Taylor
    correlation.

    Parameters
    ----------
    X : 2d list
        Data table (columns index variables, rows index samples)

    Returns
    -------
      : float
          correlation score
    '''
    Xt = list(zip(*X))

    R = [[None] * len(Xt)] * len(Xt)
    for i, xi in enumerate(Xt):
        for j, xj in enumerate(Xt):
            Xij = list(zip(*(xi, xj)))
            R[i][j] = pearson_correlation(Xij)
    lambdas = list(np.linalg.eigvals(R))
    return statistics.stdev(lambdas) / math.sqrt(len(Xt))

def trencevski_malceski_correlation(X, Y):
    '''
    Naive implementation of Trencevski-Malceski
    correlation.

    Parameters
    ----------
    X : 2d list
        Data table (columns index variables, rows index samples)
    Y : 2d list
        Data table (columns index variables, rows index samples)

    Returns
    -------
      : float
          correlation score

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> X = np.random.normal(size=1000).reshape(100,10)
    >>> X = [[float(j) for j in i] for i in X]
    >>> Y = np.random.normal(size=1000).reshape(100,10)
    >>> Y = [[float(j) for j in i] for i in Y]
    >>> trencevski_malceski_correlation(X, Y)
    3.1886981411744896e-08
    '''
    Xt = list(zip(*X))
    Yt = list(zip(*Y))
    numerator = matmult(Xt,Y)
    numerator = determinant(numerator)
    denominator = determinant(matmult(Xt, X))
    denominator *= determinant(matmult(Yt, Y))
    denominator = math.sqrt(denominator)
    return numerator / denominator

def wang_zheng_correlation(X):
    '''
    Naive implementation of Wang-Zheng
    correlation.

    Parameters
    ----------
    X : 2d list
        Data table (columns index variables, rows index samples)

    Returns
    -------
      : float
          correlation score

    Examples
    --------
    >>> import numpy as np
    >>> X = np.arange(100).reshape(10,10)
    >>> X = [[float(j) for j in i] for i in X]
    >>> wang_zheng_correlation(X)
    1.0
    '''
    Xt = list(zip(*X))

    R = [[None] * len(Xt)] * len(Xt)
    for i, xi in enumerate(Xt):
        for j, xj in enumerate(Xt):
            Xij = list(zip(*(xi, xj)))
            R[i][j] = pearson_correlation(Xij)
    return 1 - determinant(R)

def rank_simple(vector):
    return sorted(range(len(vector)), key=vector.__getitem__)

def rankdata(a, method='average'):
    '''
    Computes the rank of a list of numbers.


    https://stackoverflow.com/a/30801799/4348400
    '''
    n = len(a)
    ivec=rank_simple(a)
    svec=[a[rank] for rank in ivec]
    sumranks = 0
    dupcount = 0
    newarray = [0]*n
    for i in range(n):
        sumranks += i
        dupcount += 1
        if i==n-1 or svec[i] != svec[i+1]:
            for j in range(i-dupcount+1,i+1):
                if method=='average':
                    averank = sumranks / float(dupcount) + 1
                    newarray[ivec[j]] = averank
                elif method=='max':
                    newarray[ivec[j]] = i+1
                elif method=='min':
                    newarray[ivec[j]] = i+1 -dupcount+1
                else:
                    raise ValueError('Unsupported method')

            sumranks = 0
            dupcount = 0

    return newarray

def dot(x,y):
    '''
    Dot product between two vectors.

    Parameters
    ------
    x : 1d list
        Data vector
    y : 1d list
        Data vector

    Returns
    ------
      : float
          Vector dot product
    '''
    result = 0.0
    for xi, yi in zip(x,y):
        result += xi * yi
    return result

def matmult(a,b):
    '''
    Matrix multiplication.

    Parameters
    ------
    a : 2d list
        Data matrix
    b : 2d list
        Data matrix

    Returns
    ------
      : float
          Matrix product
    
    https://stackoverflow.com/questions/10508021/matrix-multiplication-in-pure-python
    '''
    zip_b = zip(*b)
    zip_b = list(zip_b)
    return [[sum(ele_a*ele_b for ele_a, ele_b in zip(row_a, col_b)) 
             for col_b in zip_b] for row_a in a]

def determinant(X):
    '''
    Determinant of a matrix.

    Parameters
    ------
    X : 2d list
        Data table (columns index variables, rows index samples)

    Returns
    ------
      : float
          determinant
    '''
    n = len(X)
    M = deepcopy(X)
 
    for k in range(n):
        for i in range(k+1,n):
            if M[k][k] == 0:
                M[k][k] = 1.0e-18
            s = M[i][k] / M[k][k]
            for j in range(n): 
                M[i][j] = M[i][j] - s * M[k][j]

    product = 1.0
    for i in range(n):
        product *= M[i][i] 
 
    return product
