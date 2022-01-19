import sympy
import numpy as np

def mean(f, t, a, b):
    '''
    Symbolically computes the definite
    integral representing the mean value of a
    function using uniform probability measure.

    Parameters
    ----------
    f : SymPy expression.
        Function to be integrated.
    t : `sympy.core.symbol.Symbol`.
        Independent variable to be integrated with respect to.
    a : Undefined.
        Lower bound of integration.
    b : Undefined.
        Upper bound of integration.

    Returns
    -------
    result: SymPy expression.
        Definite mean of function over given interval.
    '''
    f = sympy.sympify(f)
    result = 1/(b-a) * sympy.integrate(f, (t,a,b))
    return result

def std(f, t, a, b):
    '''
    Symbolically computes the definite
    integral representing the standard deviation of a
    function using uniform probability measure.

    Parameters
    ----------
    f : SymPy expression.
        Function to be integrated.
    t : `sympy.core.symbol.Symbol`.
        Independent variable to be integrated with respect to.
    a : Undefined.
        Lower bound of integration.
    b : Undefined.
        Upper bound of integration.

    Returns
    -------
    result: SymPy expression.
        Definite standard deviation of function over given interval.
    '''
    f = (sympy.sympify(f) - mean(f, t, a, b))**2
    return sympy.sqrt(1/(b-a) * sympy.integrate(f, (t,a,b)))

def pdev(f, t, a, b, p=2):
    '''
    Symbolically computes the definite
    integral representing the Minkowski deviation of
    order p of a function using uniform probability measure.

    Parameters
    ----------
    f : SymPy expression.
        Function to be integrated.
    t : `sympy.core.symbol.Symbol`.
        Independent variable to be integrated with respect to.
    a : Undefined.
        Lower bound of integration.
    b : Undefined.
        Upper bound of integration.

    Returns
    -------
    result : SymPy expression.
        Definite Minkowski deviation of order p of function over given interval.
    '''
    f = (sympy.Abs(sympy.sympify(f) - mean(f, t, a, b)))**p
    return (1/(b-a) * sympy.integrate(f, (t,a,b)))**(1/p)

def partial_galtonian(F, var, order):
    '''
    Computes the partial Galtonian of a
    given order with respect to a given
    variable.

    Parameters
    ----------
    F : array-like[SymPy expressions]
        Operand functions of a given variable.
    var : SymPy Symbol
        Independent parameter for differentiation/integration.
    order : int
        Order of the partial Galtonian.

    Returns
    -------
        : SymPy expression.

    Examples
    --------
    >>> t = sympy.Symbol('t')
    >>> F = [t**i for i in range(1,4)]
    >>> partial_galtonian(F, t, 2)
    6*t**3
    '''
    if not isinstance(order, int):
        raise ValueError('Order parameter must be an integer.')
    
    if order == 0:
        return np.prod(F)
    elif order > 0:
        result = 1
        for f in F:
            result *= sympy.diff(f, *(var,)*order)
        return result
    else:
        result = 1
        for j, f in enumerate(F):
            fi = f
            for i in range(-order):
                fi = sympy.integrate(fi, var) + sympy.Symbol('C_{%i,%i}' % (j,i))
            result *= fi
        return result

def multi_partial_galtonian(F, Vars, order):
    '''
    Computes the partial multi-Galtonian of a
    given order with respect to a given collection
    of variables.

    Parameters
    ----------
    F : array-like[SymPy expressions]
        Operand functions of a given variable.
    Var : array-like[SymPy Symbols]
        Independent parameters for differentiation/integration.
    order : int
        Order of the partial multi-Galtonian.

    Returns
    -------
        : SymPy expression.

    Examples
    --------
    >>> t1, t2 = sympy.var('t1 t2')
    >>> F = [(t1+i)**(t2+i) for i in range(1,4)]
    >>> multi_partial_galtonian(F, [t1, t2], 0)
    (t1 + 1)**(2*t2 + 2)*(t1 + 2)**(2*t2 + 4)*(t1 + 3)**(2*t2 + 6)
    '''
    result = 1
    for var in Vars:
        result *= partial_galtonian(F, var, order)
    return result

if __name__ == "__main__":
##    t = sympy.Symbol('x')
##    tau = sympy.Symbol('\\tau')
##    w = sympy.Symbol('w')
##    p = sympy.Symbol('p')
##    result = sympy.simplify(pdev('x', t, tau - w, tau + w,sympy.sympify('3')))
##    print(sympy.latex(sympy.simplify(sympy.expand(result))))
    t = sympy.Symbol('t')
    F = [sympy.Symbol(f'a{j}') / 2 * t **2 + sympy.Symbol(f'v{j}')*t+sympy.Symbol(f's{j}') for j in range(1,4)]
