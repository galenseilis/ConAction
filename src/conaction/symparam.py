"""
The SymParam module is provides functions for symbolically computing
instantiations of the Trinity of Covariation.

.. warning::
    Sometimes the evalf() function should be used to complete
    the calculation to a floating point value. This will often depend
    on the functions being integrated and how SymPy attempts to
    handle them.
"""

import numpy as np
import sympy


def mean(f, t, a, b):
    """
    Symbolically computes the definite
    integral representing the mean value of a
    function using uniform probability measure.

    Parameters
    ----------
    f : SymPy expression.
        Function to be integrated.
    t : `sympy.core.symbol.Symbol`.
        Independent parameter for integration.
    a : Undefined.
        Lower bound of integration.
    b : Undefined.
        Upper bound of integration.

    Returns
    -------
    result: SymPy expression.
        Definite mean of function over given interval.

    Examples
    --------
    >>> x = sympy.Symbol('x')
    >>> mean(x ** 2, x, -2, 2)
    1.33333333333333
    """
    f = sympy.sympify(f)
    result = 1 / (b - a) * sympy.integrate(f, (t, a, b))
    return result


def standard_deviation(f, t, a, b):
    """
    Symbolically computes the definite
    integral representing the standard deviation of a
    function using uniform probability measure.

    Parameters
    ----------
    f : SymPy expression.
        Function to be integrated.
    t : `sympy.core.symbol.Symbol`.
        Independent parameter for integration.
    a : Undefined.
        Lower bound of integration.
    b : Undefined.
        Upper bound of integration.

    Returns
    -------
    result: SymPy expression.
        Definite standard deviation of function over given interval.

    Examples
    --------
    >>> x = sympy.Symbol('x')
    >>> standard_deviation(x**2, x, 0, 1)
    0.298142396999972
    """
    f = (sympy.sympify(f) - mean(f, t, a, b)) ** 2
    return sympy.sqrt(1 / (b - a) * sympy.integrate(f, (t, a, b)))


def nightingale_deviation(f, var, a, b, p=2):
    """
    Symbolically computes the definite
    integral representing the Nightingale deviation of
    order p of a function using uniform probability measure.

    Parameters
    ----------
    f : SymPy expression.
        Function to be integrated.
    var : `sympy.core.symbol.Symbol`.
        Independent parameter for integration.
    a : Undefined.
        Lower bound of integration.
    b : Undefined.
        Upper bound of integration.

    Returns
    -------
    result : SymPy expression.
        Definite Nightingale deviation of order p of function over given interval.

    Examples
    --------
    >>> x = sympy.Symbol('x')
    >>> nightingale_deviation(x ** 2, x, 0, 1, 1).evalf()
    0.256600174703415
    """
    result = mean(f, var, a, b)
    result = f - result
    result = sympy.Abs(result)
    result = sympy.Pow(result, p)
    result = sympy.integrate(result, (var, a, b))
    result = 1 / (b - a) * result
    result = sympy.Pow(result, 1 / p)
    return result


def covariance(F, var, a, b):
    """
    Symbolically computes the multilinear covariance of a
    collection of functions over a given interval by integrating
    over a shared parameter.

    Parameters
    ----------
    F : array-like[SymPy expressions]
        Sequence of functions to compute covariance upon.
    var : SymPy Symbol
        Independent parameter for integration.
    a : Undefined.
        Lower bound of integration.
    b : Undefined.
        Upper bound of integration.

    Returns
    -------
    result : SymPy expression.
        Definite multilinear covariance.

    References
    ----------
    .. "Covariance.", https://en.wikipedia.org/wiki/Covariance
    .. "Mixed moment.", https://en.wikipedia.org/wiki/Moment_(mathematics)#Mixed_moments

    Examples
    --------
    >>> x = sympy.Symbol('x')
    >>> F = [x ** (i+1) for i in range(2)]
    >>> covariance(F, x, 0, 1)
    0.0833333333333333
    """
    result = [f - mean(f, var, a, b) for f in F]
    result = np.prod(result)
    result = sympy.integrate(result, (var, a, b))
    result = 1 / (b - a) * result
    return result

def nightingale_covariance(F, var, a, b, p):
    """
    Symbolically computes the Nightingale covariance of a
    collection of functions over a given interval by integrating
    over a shared parameter.

    Parameters
    ----------
    F : array-like[SymPy expressions]
        Sequence of functions to compute Nightingale covariance upon.
    var : SymPy Symbol
        Independent parameter for integration.
    a : Undefined.
        Lower bound of integration.
    b : Undefined.
        Upper bound of integration.

    Returns
    -------
    result : SymPy expression.
        Definite Nightingale covariance.

    References
    ----------
    .. "Covariance.", https://en.wikipedia.org/wiki/Covariance
    .. "Mixed moment.", https://en.wikipedia.org/wiki/Moment_(mathematics)#Mixed_moments

    Examples
    --------
    >>> x = sympy.Symbol('x')
    >>> F = [x ** (i+1) for i in range(2)]
    >>> nightingale_covariance(F, x, 0, 1, 2)
    0.115010926557059
    """
    result = [f - mean(f, var, a, b) for f in F]
    result = sympy.Abs(np.prod(result)) ** p
    result = sympy.integrate(result, (var, a, b))
    result = 1 / (b - a) * result
    result = result ** (1 / p)
    return result

def nightingale_correlation(F, var, a, b, p):
    """
    Symbolically computes the Nightingale
    correlation coefficient on a collection of functions over a given
    interval by integrating over a shared parameter.

    Parameters
    ----------
    F : array-like[SymPy expressions]
        Sequence of functions to compute correlation coefficient upon.
    var : SymPy Symbol
        Independent parameter for integration.
    a : Undefined.
        Lower bound of integration.
    b : Undefined.
        Upper bound of integration.

    Returns
    -------
    result : SymPy expression.
        Definite Nightingale's correlation.

    References
    ----------
    .. "Correlation.", https://en.wikipedia.org/wiki/Correlation

    Examples
    --------
    >>> x = sympy.Symbol('x')
    >>> F = [x ** (i+1) for i in range(2)]
    >>> nightingale_correlation(F, x, 0, 100, 1).evalf()
    0.970246182902770
    """
    numerator = nightingale_covariance(F, var, a, b, p)
    denominator = [nightingale_deviation(f, var, a, b, p=len(F)) for f in F]
    denominator = np.prod(denominator)
    result = numerator / denominator
    return result

def pearson_correlation(F, var, a, b):
    """
    Symbolically computes the multilinear Pearson's product-moment
    correlation coefficient on a collection of functions over a given
    interval by integrating over a shared parameter.

    Parameters
    ----------
    F : array-like[SymPy expressions]
        Sequence of functions to compute correlation coefficient upon.
    var : SymPy Symbol
        Independent parameter for integration.
    a : Undefined.
        Lower bound of integration.
    b : Undefined.
        Upper bound of integration.

    Returns
    -------
    result : SymPy expression.
        Definite multilinear Pearson's correlation.

    References
    ----------
    .. "Correlation.", https://en.wikipedia.org/wiki/Correlation
    .. "Pearson correlation coefficient.", https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

    Examples
    --------
    >>> x = sympy.Symbol('x')
    >>> F = [x ** (i+1) for i in range(3)]
    >>> pearson_correlation(F, x, 0, 100).evalf()
    0.398761062646958
    """
    numerator = covariance(F, var, a, b)
    denominator = [nightingale_deviation(f, var, a, b, p=len(F)) for f in F]
    denominator = np.prod(denominator)
    result = numerator / denominator
    return result


def reflective_correlation(F, var, a, b):
    """
    Symbolically computes the multilinear reflective
    correlation coefficient on a collection of functions over a given
    interval by integrating over a shared parameter.

    Parameters
    ----------
    F : array-like[SymPy expressions]
        Sequence of functions to compute correlation coefficient upon.
    var : SymPy Symbol
        Independent parameter for integration.
    a : Undefined.
        Lower bound of integration.
    b : Undefined.
        Upper bound of integration.

    Returns
    -------
    result : SymPy expression.
        Definite multilinear reflective correlation.

    References
    ----------
    .. "Correlation.", https://en.wikipedia.org/wiki/Correlation
    .. "Circular correlation coefficient.", https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Reflective_correlation_coefficient

    Examples
    --------
    >>> x = sympy.Symbol('x')
    >>> F = [x ** (i+1) for i in range(2)]
    >>> reflective_correlation(F, x, 0, 100).evalf()
    0.968245836551854
    """
    numerator = np.prod(F)
    numerator = 1 / (b - a) * sympy.integrate(numerator, (var, a, b))
    denominator = [sympy.Abs(f) for f in F]
    denominator = [sympy.Pow(f, len(F)) for f in denominator]
    denominator = [1 / (b - a) * sympy.integrate(f, (var, a, b)) for f in denominator]
    denominator = [sympy.Pow(f, 1 / len(F)) for f in denominator]
    denominator = np.prod(denominator)
    result = numerator / denominator
    return result


def circular_correlation(F, var, a, b):
    """
    Symbolically computes the multilinear circular
    correlation coefficient on a collection of functions over a given
    interval by integrating over a shared parameter.

    Parameters
    ----------
    F : array-like[SymPy expressions]
        Sequence of functions to compute correlation coefficient upon.
    var : SymPy Symbol
        Independent parameter for integration.
    a : Undefined.
        Lower bound of integration.
    b : Undefined.
        Upper bound of integration.

    Returns
    -------
    result : SymPy expression.
        Definite multilinear circular correlation.

    References
    ----------
    .. "Correlation.", https://en.wikipedia.org/wiki/Correlation
    .. "Circular correlation coefficient.", https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Circular_correlation_coefficient

    Examples
    --------
    >>> x = sympy.Symbol('x')
    >>> F = [x ** (i+1) for i in range(2)]
    >>> circular_correlation(F, x, 0, 100).evalf()
    -0.e-1
    """
    result = [sympy.sin(f - mean(f, var, a, b)) for f in F]
    result = reflective_correlation(result, var, a, b)
    return result


def signum_correlation(F, var, a, b):
    """
    Signum correlation coefficient.

    Symbolically computes the multilinear signum
    correlation coefficient on a collection of functions over a given
    interval by integrating over a shared parameter.

    This function estimates

    .. math::
        R_{\\text{sign}} \\left[ X_1, \\cdots, X_n \\right] = \\frac{\\mathbb{E} \\left[ \\prod_{j=1}^{n} \\text{sign} \\left( X_j - \\mathbb{E}[X_j] \\right) \\right]}{\\prod_{j=1}^{n} \\sqrt[n]{\\mathbb{E}\\left[ |\\text{sign} \\left( X_j - \\mathbb{E}[X_j] \\right)|^n \\right]}}

    Parameters
    ----------
    F : array-like[SymPy expressions]
        Sequence of functions to compute correlation coefficient upon.
    var : SymPy Symbol
        Independent parameter for integration.
    a : Undefined.
        Lower bound of integration.
    b : Undefined.
        Upper bound of integration.

    Returns
    -------
    result : SymPy expression.
        Definite multilinear sigum correlation.

    References
    ----------
    .. "Correlation.", https://en.wikipedia.org/wiki/Correlation

    Examples
    --------
    >>> x = sympy.Symbol('x')
    >>> F = [x ** (i+1) for i in range(2)]
    >>> signum_correlation(F, x, 0, 100)
    0.845299461620749
    """
    result = [sympy.sign(f - mean(f, var, a, b)) for f in F]
    result = reflective_correlation(result, var, a, b)
    return result


def misiak_correlation(fx, fy, F, var, a, b):
    """
    Symbolically computes the Misiak correlation coefficient
    on a collection of functions over a given interval by
    integrating over a shared parameter.

    Parameters
    ----------
    fx : SymPy expression.
        A function.
    fy : SymPy expression.
        A function.
    F : array-like[SymPy expressions]
        Sequence of functions to compute correlation coefficient upon.
    var : SymPy Symbol
        Independent parameter for integration.
    a : Undefined.
        Lower bound of integration.
    b : Undefined.
        Upper bound of integration.

    Returns
    -------
    result : SymPy expression.
        Definite misiak correlation.

    References
    ----------
    .. Misiak, Aleksander. Ryz, Alicja. 2000. "n-Inner Product Spaces and Projections.", https://www.emis.de/journals/MB/125.1/mb125_1_5.pdf

    Examples
    --------
    >>> x = sympy.Symbol('x')
    >>> fx = sympy.sqrt(x)
    >>> fy = sympy.sin(x)
    >>> F = [x, x**2]
    >>> misiak_correlation(fx, fy, F, x, 1, 2).evalf()
    -0.999698940593851
    """
    G = np.empty((len(F) + 1, len(F) + 1), dtype=object)
    G[0, 0] = sympy.integrate(fx * fy, (var, a, b))
    for i, fi in enumerate(F):
        G[0, i + 1] = sympy.integrate(fx * fi, (var, a, b))
        G[i + 1, 0] = sympy.integrate(fy * fi, (var, a, b))
        for j, fj in enumerate(F):
            G[i + 1, j + 1] = sympy.integrate(fi * fj, (var, a, b))
    numerator = sympy.Matrix(G).det()
    G[0, 0] = sympy.integrate(fx * fx, (var, a, b))
    for i, fi in enumerate(F):
        G[i + 1, 0] = sympy.integrate(fx * fi, (var, a, b))
    denominator = sympy.Matrix(G).det()
    G[0, 0] = sympy.integrate(fy * fy, (var, a, b))
    for i, fi in enumerate(F):
        G[0, i + 1] = sympy.integrate(fy * fi, (var, a, b))
        G[i + 1, 0] = sympy.integrate(fy * fi, (var, a, b))
    denominator = sympy.Matrix(G).det() * denominator
    denominator = sympy.sqrt(denominator)
    result = numerator / denominator
    return result


def taylor_correlation(F, var, a, b):
    """
    Taylor's multi-way correlation coefficient.

    Taylor 2020 defines this function to be

    .. math::

        \\frac{1}{\\sqrt{d}} \\sqrt{\\frac{1}{d-1} \\sum_{i}^{d} ( \\lambda_i -  \\bar{\\lambda})^2 }

    where :math:`d` is the number of variables, :math:`\lambda_1, \cdots, \lambda_d` are the eigenvalues of
    the correlation matrix for a given set of variables, and :math:`\\bar{\\lambda}` is the mean of those eigenvalues.

    Parameters
    ----------
    F : array-like[SymPy expressions]
        Sequence of functions to compute correlation coefficient upon.
    var : SymPy Symbol
        Independent parameter for integration.
    a : Undefined.
        Lower bound of integration.
    b : Undefined.
        Upper bound of integration.

    Returns
    -------
    result : SymPy expression.
        Definite Taylor correlation.

    Notes
    -----
    Taylor's multi-way correlation coefficient is a rescaling of the Bessel-corrected standard deviation of the
    eigenvalues of the correlation matrix of the set of variables.

    References
    ----------
    .. Taylor, BM. 2020. "A Multi-Way Correlation Coefficient", https://arxiv.org/abs/2003.02561

    Examples
    --------
    >>> x = sympy.Symbol('x')
    >>> F = [x**(i+1) for i in range(3)]
    >>> taylor_correlation(F, x, 0, 1)
    0.957378751630761
    """
    R = np.empty((len(F), len(F)), dtype=object)
    for i, fi in enumerate(F):
        for j, fj in enumerate(F):
            R[i, j] = pearson_correlation([fi, fj], var, a, b)
    eig = sympy.Matrix(R).eigenvals()
    eigenvalues = []
    for key, value in eig.items():
        eigenvalues += [key] * value
    result = np.array(eigenvalues, dtype=object)
    result = np.power(result - np.sum(result) / len(F), 2)
    result = np.sum(result) / (len(F) - 1)
    result = sympy.sqrt(result)
    result = result / np.sqrt(len(F))
    return result


def trencevski_malceski_correlation(Fx, Fy, var, a, b):
    """
    Generalized n-inner product correlation coefficient.

    Computes a correlation coefficient based
    on Trencevski and Melceski 2006.

    Parameters
    ----------
    Fx : array-like[SymPy expressions]
        Sequence of functions to compute correlation coefficient upon.
    Fy : array-like[SymPy expressions]
        Sequence of functions to compute correlation coefficient upon.
    var : SymPy Symbol
        Independent parameter for integration.
    a : Undefined.
        Lower bound of integration.
    b : Undefined.
        Upper bound of integration.

    Returns
    -------
    result : SymPy expression.
        Definite Trencevski and Melceski correlation.

    Raises
    ------
    ValueError : Fx and Fy must have the same length

    References
    ----------
    .. Trencevski, Kostadin. Malceski, Risto. 2006. "On a generalized n-inner product and the corresponding Cauchy-Schwarz inequality", https://www.researchgate.net/publication/268999118_On_a_generalized_n-inner_product_and_the_corresponding_Cauchy-Schwarz_inequality

    Examples
    --------
    >>> x = sympy.Symbol('x')
    >>> Fx = [x**(i+1) for i in range(3)]
    >>> Fy = [sympy.sin(x**(i+1)) for i in range(3)]
    >>> trencevski_malceski_correlation(Fx, Fy, x, 0, 1)
    0.994375897094607
    """
    if len(Fx) != len(Fy):
        raise ValueError("Fx and Fy must have the same length")

    G = np.empty((len(Fx), len(Fy)))
    for i, fi in enumerate(Fx):
        for j, fj in enumerate(Fy):
            G[i, j] = sympy.integrate(fi * fj, (var, a, b))
    numerator = sympy.Matrix(G).det()
    denominator = 1
    for Fk in [Fx, Fy]:
        for i, fi in enumerate(Fk):
            for j, fj in enumerate(Fk):
                G[i, j] = sympy.integrate(fi * fj, (var, a, b))
        denominator *= sympy.Matrix(G).det()
    denominator = sympy.sqrt(denominator)
    result = numerator / denominator
    return result


def wang_zheng_correlation(F, var, a, b):
    """
    Correlation coefficient due to Wang & Zheng 2014.

    This correlation coefficient is equivalent to

    .. math::

        R_{wz} \\triangleq 1 - \\det (R_{n \\times n})

    where :math:`R_{n \\times n}` is the correlation
    matrix computed on a collection of n variables. In
    other words, this correlation coefficient is the
    complement of the determinant of the correlation
    matrix.


    Parameters
    ----------
    F : array-like[SymPy expressions]
        Sequence of functions to compute correlation coefficient upon.
    var : SymPy Symbol
        Independent parameter for integration.
    a : Undefined.
        Lower bound of integration.
    b : Undefined.
        Upper bound of integration.

    Returns
    -------
    result : SymPy expression.
        Definite Wang Zheng correlation.

    Notes
    -----
    The complement of this statistic is the unsigned incorrelation coefficient.

    References
    ----------
    .. Wang, Jianji. Zheng, Nanning. 2014. "Measures of Correlation for Multiple Variables", https://arxiv.org/abs/1401.4827

    Examples
    --------
    >>> x = sympy.Symbol('x')
    >>> F = [x**(i+1) for i in range(3)]
    >>> wang_zheng_correlation(F, x, 0, 1)
    0.999722222222222
    """
    R = np.empty((len(F), len(F)))
    for i, fi in enumerate(F):
        for j, fj in enumerate(F):
            R[i, j] = pearson_correlation([fi, fj], var, a, b)
    result = sympy.Matrix(R).det()
    result = 1 - result
    return result


def partial_agnesian(F, var, order):
    """
    Computes the partial Agnesian of a
    given order with respect to a given
    variable.

    Parameters
    ----------
    F : array-like[SymPy expressions]
        Operand functions of a given variable.
    var : SymPy Symbol
        Independent parameter for differentiation/integration.
    order : int
        Order of the partial Agnesian.

    Returns
    -------
        : SymPy expression.

    Examples
    --------
    >>> t = sympy.Symbol('t')
    >>> F = [t**i for i in range(1,4)]
    >>> partial_agnesian(F, t, 2)
    0
    """
    if not isinstance(order, int):
        raise ValueError("Order parameter must be an integer.")

    if order == 0:
        return np.prod(F)
    elif order > 0:
        result = 1
        for f in F:
            result *= sympy.diff(f, *(var,) * order)
        return result
    else:
        result = 1
        for j, f in enumerate(F):
            fi = f
            for i in range(-order):
                fi = sympy.integrate(fi, var) + sympy.Symbol("C_{%i,%i}" % (j, i))
            result *= fi
        return result


def partial_multiagnesian(F, Vars, order):
    """
    Computes the partial multiagnesian
     of a given order with respect to a given collection of
     variables.

    Parameters
    ----------
    F : array-like[SymPy expressions]
        Operand functions of a given variable.
    Vars : array-like[SymPy Symbols]
        Independent parameters for differentiation/integration.
    order : int
        Order of the partial multiagnesian.

    Returns
    -------
        : SymPy expression.

    Examples
    --------
    >>> t1, t2 = sympy.var('t1 t2')
    >>> F = [(t1+i)**(t2+i) for i in range(1,4)]
    >>> partial_multiagnesian(F, [t1, t2], 0)
    (t1 + 1)**(2*t2 + 2)*(t1 + 2)**(2*t2 + 4)*(t1 + 3)**(2*t2 + 6)
    """
    result = 1
    for var in Vars:
        result *= partial_differential_covariance(F, var, order)
    return result
