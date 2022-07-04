'''
This submodule contains functions for the numerical integration
of parametric objects (e.g. curves and surfaces). It is built on
top of NumPy and SciPy for efficient numerical integration over
multiple bounds. Additionally, some functions use Pathos to
parallelize the processing of expressions involving separate
integration steps.
'''

import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
from scipy import integrate
from scipy.linalg import svdvals


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

def root_moment(f, I, p=2, *args, **kwargs):
    '''
    Numerically computes the definite
    integral representing the root moment value of a
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
    >>> root_moment(F, I)
    1.0801234497346432

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
        return np.power(np.abs(f(*intargs)), p)
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
    >>> covariance(F, I)
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

def product_moment(F, I, *args, **kwargs):
    '''
    Numerically computes the definite
    integral representing the mixed uncentered product moment value of a
    collection of functions using uniform probability measure.

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
    >>> product_moment(F, I)
    1.1666666666666665

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
        poolf = lambda f: f(*intargs)
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

def reflective_correlation(F, I, *args, **kwargs):
    '''
    Numerically computes the definite
    integral representing the multilinear reflective correlation
    value of a collection of functions using uniform probability measure. The
    reflective correlation coefficient here has
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
    >>> reflective_correlation(F, I)
    1.0

    .. warning::
        The length and order of `I` must correspond
        to the variables in F.
    '''
    numerator = product_moment(F, I, *args, **kwargs)
    p = Pool()
    poolf = lambda f: root_moment(f, I, p=len(F), *args, **kwargs)
    denominator = p.map(poolf, F)
    denominator = np.prod(denominator)
    return numerator / denominator

def circular_correlation(F, I, *args, **kwargs):
    '''
    Numerically computes the definite
    integral representing the multilinear circular correlation
    of a collection of unctions using uniform probability measure. The
    circular correlation coefficient here has
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
    >>> circular_correlation(F, I)
    1.0

    .. warning::
        The length and order of `I` must correspond
        to the variables in F.
    '''
    means = [mean(f, I, *args, **kwargs) for f in F]
    T = [lambda *x: np.sin(fi(*x) - mi) for fi, mi in zip(F,means)]
    return reflective_correlation(T, I, *args, **kwargs)

def signum_correlation(F, I, *args, **kwargs):
    '''
    Numerically computes the definite
    integral representing the multilinear circular correlation
    of a collection of unctions using uniform probability measure. The
    circular correlation coefficient here has
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
    >>> signum_correlation(F, I)
    1.0

    .. warning::
        The length and order of `I` must correspond
        to the variables in F.
    '''
    means = [mean(f, I, *args, **kwargs) for f in F]
    T = [lambda *x: np.sign(fi(*x) - mi) for fi, mi in zip(F,means)]
    return reflective_correlation(T, I, *args, **kwargs)

def taylor_correlation(F, I, *args, **kwargs):
    '''
    Numerically computes Taylor's multi-way correlation
    coefficient for a given collcetion of functions using
    definite integration.

    Taylor 2020 defines this function to be

    .. math::

        \\frac{1}{\\sqrt{d}} \\sqrt{\\frac{1}{d-1} \\sum_{i}^{d} ( \\lambda_i -  \\bar{\\lambda})^2 }

    where :math:`d` is the number of variables, :math:`\lambda_1, \cdots, \lambda_d` are the eigenvalues of
    the correlation matrix for a given set of variables, and :math:`\\bar{\\lambda}` is the mean of those eigenvalues.

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

    Notes
    -----
    Taylor's multi-way correlation coefficient is a rescaling of the Bessel-corrected standard deviation of the
    eigenvalues of the correlation matrix of the set of variables.

    References
    ----------
    .. [1] Taylor, BM. 2020. "A Multi-Way Correlation Coefficient", https://arxiv.org/abs/2003.02561

    Examples
    --------
    >>> F = [lambda x: x**(i+1) for i in range(3)]
    >>> I = [(0,1)]
    >>> taylor_correlation(F, I)
    1.0
    '''
    R = np.empty((len(F), len(F)))
    for i, fi in enumerate(F):
        for j, fj in enumerate(F):
            R[i, j] = pearson_correlation([fi, fj], I, *args, **kwargs)
    result = svdvals(R)
    result = result - np.mean(result)
    result = np.power(result, 2)
    result = np.sum(result) / (len(F) - 1)
    result = np.sqrt(result)
    result /= np.sqrt(len(F))
    return result
    

if __name__ == "__main__":
    F = [lambda x: x**(i+1) for i in range(3)]
    I = [(0,1)]
    print(taylor_correlation(F, I))
