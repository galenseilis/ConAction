'''
This submodule contains functions for the numerical integration
of parametric objects (e.g. curves and surfaces). It is built on
top of NumPy and SciPy for efficient numerical integration over
multiple bounds. Additionally, some functions use Pathos to
parallelize the processing of expressions involving separate
integration steps.
'''

import numpy as np
from scipy import integrate
from pathos.multiprocessing import ProcessingPool as Pool

def mean(f, I, *args, **kwargs):
    '''
    Numerically computes the definite
    integral representing the mean value of a
    function using uniform probability measure.

    Parameters
    ----------
    f: function.
        Function to be integrated.
    I: array-like
        Integration bounds.

    Returns
    -------
    result: float
        Definite integral.

    Examples
    --------
    >>> F = lambda x,y: x+y
    >>> I = [(0,1)]*2
    >>> mean(F, I)
    1.0

    .. warning::
        The length and order of `I` must correspond
        to the variables in f.
    '''
    lower = np.array(I)[:,0]
    upper = np.array(I)[:,1]
    scale = upper - lower
    scale = np.prod(scale)
    result = integrate.nquad(f, I, *args, **kwargs)
    result = result[0]
    result /= scale
    return result

def minkowski_deviation(f, I, p=2, *args, **kwargs):
    '''
    Numerically computes the definite
    integral representing the Minkowski deviation value of a
    function using uniform probability measure. The Minkowski
    deviation of order 2 is t he standard deviation.

    Parameters
    ----------
    f: function.
        Function to be integrated.
    I: array-like
        Integration bounds.

    Returns
    -------
    result: float
        Definite integral.

    Examples
    --------
    >>> F = lambda x,y: x+y
    >>> I = [(0,1)]*2
    >>> minkowski_deviation(F, I)
    0.408248290463863

    .. warning::
        The length and order of `I` must correspond
        to the variables in f.

    .. warning::
        A sufficiently large input value for p can
        result numerical issues such as `arithmetic underflow underflow <https://en.wikipedia.org/wiki/Arithmetic_underflow>`_.
    '''
    lower = np.array(I)[:,0]
    upper = np.array(I)[:,1]
    scale = np.prod(upper - lower)
    def integrand(*intargs):
        # Is computing the mean 'here' a slow-down?
        return np.power(np.abs(f(*intargs) - mean(f, I, *args, **kwargs)), p)
    result = integrate.nquad(integrand, I, *args, **kwargs)
    result = result[0]
    result /= scale
    result = np.power(result, 1/p)
    return result

def covariance(F, I, *args, **kwargs):
    '''
    Numerically computes the definite
    integral representing the multilinear covariance value of a
    function using uniform probability measure. The covariance
    here is generalized to include mixed-centered product
    moments.

    Parameters
    ----------
    F: array-like[function].
        Functions to be integrated.
    I: array-like
        Integration bounds.

    Returns
    -------
    result: float
        Definite integral.

    Examples
    --------
    >>> F = [lambda x,y: x+y for i in range(2)]
    >>> I = [(0,1)]*2
    >>> pearson_correlation(F, I)
    0.16666666666666666

    .. warning::
        The length and order of `I` must correspond
        to the variables in F.
    '''
    lower = np.array(I)[:,0]
    upper = np.array(I)[:,1]
    scale = np.prod(upper - lower)
    def integrand(*intargs):
        p = Pool()
        # Is computing the mean 'here' a slow-down?
        poolf = lambda f: f(*intargs) - mean(f, I, *args, **kwargs)
        return np.prod(p.map(poolf, F))
    result = integrate.nquad(integrand, I, *args, **kwargs)
    result = result[0]
    result /= scale
    return result

def pearson_correlation(F, I, *args, **kwargs):
    '''
    Numerically computes the definite
    integral representing the multilinear Pearson product-moment
    value of a function using uniform probability measure. The
    Pearson's product-moment correlation coefficient here has
    been generalized to allow more than two variables.

    Parameters
    ----------
    F: array-like[function].
        Functions to be integrated.
    I: array-like
        Integration bounds.

    Returns
    -------
    result: float
        Definite integral.

    Examples
    --------
    >>> F = [lambda x,y: x+y for i in range(2)]
    >>> I = [(0,1)]*2
    >>> pearson_correlation(F, I)
    1.0

    .. warning::
        The length and order of `I` must correspond
        to the variables in F.
    '''
    numerator = covariance(F, I, *args, **kwargs)
    p = Pool()
    poolf = lambda f: minkowski_deviation(f, I, p=len(F), *args, **kwargs)
    denominator = p.map(poolf, F)
    denominator = np.prod(denominator)
    return numerator / denominator

if __name__ == "__main__":
    F = lambda x,y: x+y
    I = [(0,1)]*2
    print(mean(F, I))
