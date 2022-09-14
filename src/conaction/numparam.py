"""
This submodule contains functions for the numerical integration
of parametric objects (e.g. curves and surfaces). It is built on
top of NumPy and SciPy for efficient numerical integration over
multiple bounds. Additionally, some functions use Pathos to
parallelize the processing of expressions involving separate
integration steps.
"""

import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
from scipy import integrate
from scipy.linalg import svdvals


def mean(f, I, *args, **kwargs):
    """
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
    """
    lower = np.array(I)[:, 0]
    upper = np.array(I)[:, 1]
    scale = upper - lower
    scale = np.prod(scale)
    result = integrate.nquad(f, I, *args, **kwargs)
    result = result[0]
    result /= scale
    return result


def nightingale_deviation(f, I, p=2, *args, **kwargs):
    """
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
    >>> nightingale_deviation(F, I)
    0.408248290463863

    .. warning::
        The length and order of `I` must correspond
        to the variables in f.

    .. warning::
        A sufficiently large input value for p can
        result numerical issues such as `arithmetic underflow underflow <https://en.wikipedia.org/wiki/Arithmetic_underflow>`_.
    """
    lower = np.array(I)[:, 0]
    upper = np.array(I)[:, 1]
    scale = np.prod(upper - lower)

    def integrand(*intargs):
        # Is computing the mean 'here' a slow-down?
        return np.power(np.abs(f(*intargs) - mean(f, I, *args, **kwargs)), p)

    result = integrate.nquad(integrand, I, *args, **kwargs)
    result = result[0]
    result /= scale
    result = np.power(result, 1 / p)
    return result

def standard_deviation(f, I, *args, **kwargs):
    """
    Numerically computes the definite integral representing the standard deviation value
    of a function using uniform probability measure.

    Parameters
    ----------
    f: function.
        Function to be integrated.
    I: array-like
        Integration bounds.

    Returns
    -------
    result: float
        Detinite integral.

    Examples
    --------
    >>> F = lambda x,y: x+y
    >>> I = [(0,1)]*2
    >>> nightingale_deviation(F, I)
    0.408248290463863

    .. warning::
        The length and order of `I` must correspond
        to the variables in f.

    .. warning::
        A sufficiently large input value for p can
        result numerical issues such as `arithmetic underflow underflow <https://en.wikipedia.org/wiki/Arithmetic_underflow>`_.
    """
    return nightingale_deviation(f, I, p=2, *args, **kwargs)


def root_moment(f, I, p=2, *args, **kwargs):
    """
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
    """
    lower = np.array(I)[:, 0]
    upper = np.array(I)[:, 1]
    scale = np.prod(upper - lower)

    def integrand(*intargs):
        # Is computing the mean 'here' a slow-down?
        return np.power(np.abs(f(*intargs)), p)

    result = integrate.nquad(integrand, I, *args, **kwargs)
    result = result[0]
    result /= scale
    result = np.power(result, 1 / p)
    return result


def covariance(F, I, *args, **kwargs):
    """
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
    """
    lower = np.array(I)[:, 0]
    upper = np.array(I)[:, 1]
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

def misiak_correlation(fx, fy, F, I, *args, **kwargs):
    """
    Numerically computes the Misiak correlation coefficient
    on a collection of functions over a given interval by
    integrating over a shared parameter.
    
    Parameters
    ----------
    fx : function.
        A function.
    fy : function.
        A function.
    F : array-like[functions]
        Sequence of functions to compute correlation coefficient upon.
    I: array-like
        Integration bounds.
        
    Returns
    -------
    result : SymPy expression.
        Definite misiak correlation.
        
    References
    ----------
    .. Misiak, Aleksander. Ryz, Alicja. 2000. "n-Inner Product Spaces and Projections.", https://www.emis.de/journals/MB/125.1/mb125_1_5.pdf
    
    Examples
    --------
    >>> f1 = lambda x: np.exp(x)
    >>> f2 = lambda x: np.sin(x) + np.exp(x)
    >>> F = [lambda x: x**2, lambda x: np.cos(x)+x**3 ]
    >>> I = [(0, np.pi)]
    >>> misiak_correlation(f1, f2, F, I)
    0.7383533743159028
    """
    G = np.zeros((len(F) + 1, len(F) + 1))
    G[0, 0] = integrate.nquad(lambda x,y: fx(x) * fy(y), I*2, *args, **kwargs)[0]
    for i, fi in enumerate(F):
        G[0, i + 1] = integrate.nquad(lambda x,y: fx(x) * fi(y), I*2, *args, **kwargs)[0]
        G[i + 1, 0] = integrate.nquad(lambda x,y: fy(x) * fi(y), I*2, *args, **kwargs)[0]
        for j, fj in enumerate(F):
            G[i + 1, j + 1] = integrate.nquad(lambda x,y: fi(x) * fj(y), I*2, *args, **kwargs)[0]
    numerator = np.linalg.det(G)
    G[0, 0] = integrate.nquad(lambda x,y: fx(x) * fx(y), I*2, *args, **kwargs)[0]
    for i, fi in enumerate(F):
        G[i + 1, 0] = integrate.nquad(lambda x,y: fx(x) * fi(y), I*2, *args, **kwargs)[0]
    denominator = np.linalg.det(G)
    G[0, 0] = integrate.nquad(lambda x,y: fy(x) * fy(y), I*2, *args, **kwargs)[0]
    for i, fi in enumerate(F):
        G[0, i + 1] = integrate.nquad(lambda x,y: fy(x) * fi(y), I*2, *args, **kwargs)[0]
        G[i + 1, 0] = integrate.nquad(lambda x,y: fy(x) * fi(y), I*2, *args, **kwargs)[0]
    denominator = np.linalg.det(G) * denominator
    denominator = np.sqrt(denominator)
    result = numerator / denominator
    return result

def nightingale_covariance(F, I, p=1, *args, **kwargs):
    """
    Numerically computes the definite
    integral representing the Nightingale covariance value of a
    function using uniform probability measure.

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
    >>> F = [lambda x: x for i in range(2)]
    >>> I = [(0,1)]
    >>> nightingale_covariance(F, I)
    0.08333333333333333

    .. warning::
        The length and order of `I` must correspond
        to the variables in F.
    """
    lower = np.array(I)[:, 0]
    upper = np.array(I)[:, 1]
    scale = np.prod(upper - lower)

    def integrand(*intargs):
        pool = Pool()
        # Is computing the mean 'here' a slow-down?
        poolf = lambda f: np.power(np.abs(f(*intargs) - mean(f, I, *args, **kwargs)), p)
        return np.prod(pool.map(poolf, F))

    result = integrate.nquad(integrand, I, *args, **kwargs)
    result = result[0]
    result /= scale
    result = np.power(result, 1/p)
    return result

def nightingale_correlation(F, I, *args, **kwargs):
    """
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
    """
    numerator = nightingale_covariance(F, I, *args, **kwargs)
    pool = Pool()
    poolf = lambda f: nightingale_deviation(f, I, p=len(F), *args, **kwargs)
    denominator = pool.map(poolf, F)
    denominator = np.prod(denominator)
    return numerator / denominator

def partial_agnesian(F, I=[(0,1)], order=1, steps=50, *args, **kwargs):
    """
    Computes the partial Agnesian of a
    given order with respect to a given
    variable.
    Parameters
    ----------
    F : array-like[function]
        Operand functions of a given variable.
    I: array-like
        Integration bounds.
    order : int
        Order of the partial Agnesian.
    Returns
    -------
        : float
    Examples
    --------
    >>> F = [lambda x : x, lambda x: x]
    >>> I = [(0, 2)]
    >>> partial_agnesian(F, I, order=-2)
    16.0
    """
    if not isinstance(order, int):
        raise ValueError("Order parameter must be an integer.")

    t = np.linspace(I[0][0], I[0][1], num=steps)

    if order == 0:
        result = [f(t) for f in F]
        result = np.array(result).T
        result = np.prod(result, axis=1)
        return result
    elif order > 0:
        dt = (t[-1] - t[0]) / steps
        X = [f(t) for f in F]
        X = np.array(X).T
        for i in range(order):
            X = np.diff(X, axis=0) / dt
        result = np.prod(X, axis=1)
        return result
    elif order == -1:
        result = 1
        for j, f in enumerate(F):
            fi = f
            fi = integrate.nquad(fi, I, *args, **kwargs)[0]
            result *= fi
        return result
    elif order == -2:
        result = 1
        for j, f in enumerate(F):
            fi = f
            fi = integrate.nquad(lambda x: integrate.nquad(lambda x : fi(x), I)[0], I)[0]
            result *= fi
        return result
    elif order == -3:
        result = 1
        for j, f in enumerate(F):
            fi = f
            fi = integrate.nquad(lambda x: integrate.nquad(lambda x: integrate.nquad(lambda x : fi(x), I)[0], I)[0], I)[0]
            result *= fi
        return result
    else:
        raise NotImplemented('Orders below -3 are not available yet.')

def product_moment(F, I, *args, **kwargs):
    """
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
    """
    lower = np.array(I)[:, 0]
    upper = np.array(I)[:, 1]
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
    """
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
    """
    numerator = covariance(F, I, *args, **kwargs)
    p = Pool()
    poolf = lambda f: nightingale_deviation(f, I, p=len(F), *args, **kwargs)
    denominator = p.map(poolf, F)
    denominator = np.prod(denominator)
    return numerator / denominator


def reflective_correlation(F, I, *args, **kwargs):
    """
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
    """
    numerator = product_moment(F, I, *args, **kwargs)
    p = Pool()
    poolf = lambda f: root_moment(f, I, p=len(F), *args, **kwargs)
    denominator = p.map(poolf, F)
    denominator = np.prod(denominator)
    return numerator / denominator


def circular_correlation(F, I, *args, **kwargs):
    """
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
    """
    means = [mean(f, I, *args, **kwargs) for f in F]
    T = [lambda *x: np.sin(fi(*x) - mi) for fi, mi in zip(F, means)]
    return reflective_correlation(T, I, *args, **kwargs)


def signum_correlation(F, I, *args, **kwargs):
    """
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
    """
    means = [mean(f, I, *args, **kwargs) for f in F]
    T = [lambda *x: np.sign(fi(*x) - mi) for fi, mi in zip(F, means)]
    return reflective_correlation(T, I, *args, **kwargs)


def taylor_correlation(F, I, *args, **kwargs):
    """
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
    """
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

def trencevski_malceski_correlation(Fx, Fy, I, *args, **kwargs):
    """
    Generalized n-inner product correlation coefficient.
    Computes a correlation coefficient based
    on Trencevski and Melceski 2006.
    
    Parameters
    ----------
    Fx : array-like[function]
        Sequence of functions to compute correlation coefficient upon.
    Fy : array-like[function]
        Sequence of functions to compute correlation coefficient upon.
    I: array-like
        Integration bounds.
        
    Returns
    -------
    result : float
        Definite Trencevski and Melceski correlation.
        
    Raises
    ------
    ValueError : Fx and Fy must have the same length
    
    References
    ----------
    .. Trencevski, Kostadin. Malceski, Risto. 2006. "On a generalized n-inner product and the corresponding Cauchy-Schwarz inequality", https://www.researchgate.net/publication/268999118_On_a_generalized_n-inner_product_and_the_corresponding_Cauchy-Schwarz_inequality

    Examples
    --------
    >>> Fx = [lambda x : x**4, lambda x: x / 3]
    >>> Fy = [lambda x: np.exp(x), lambda x: x ** 3]
    I = [(0, 2)]
    >>> trencevski_malceski_correlation(Fx, Fy, I)
    0.7071067811865445
    """
    if len(Fx) != len(Fy):
        raise ValueError("Fx and Fy must have the same length")

    G = np.empty((len(Fx), len(Fy)))
    for i, fi in enumerate(Fx):
        for j, fj in enumerate(Fy):
            G[i, j] = integrate.nquad(lambda x,y: fi(x) * fj(y), I*2, *args, **kwargs)[0]
    numerator = np.linalg.det(G)
    denominator = 1
    for Fk in [Fx, Fy]:
        for i, fi in enumerate(Fk):
            for j, fj in enumerate(Fk):
                G[i, j] = integrate.nquad(lambda x,y: fi(x) * fj(y), I*2, *args, **kwargs)[0]
        denominator *= np.linalg.det(G)
    denominator = np.sqrt(denominator)
    result = numerator / denominator
    return result
