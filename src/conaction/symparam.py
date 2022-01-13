import sympy
import numpy as np

def mean(f, t, a, b):
    f = sympy.sympify(f)
    result = 1/(b-a) * sympy.integrate(f, (t,a,b))
    return result

def std(f, t, a, b):
    f = (sympy.sympify(f) - mean(f, t, a, b))**2
    return sympy.sqrt(1/(b-a) * sympy.integrate(f, (t,a,b)))

def pdev(f, t, a, b, p=2):
    f = (sympy.Abs(sympy.sympify(f) - mean(f, t, a, b)))**p
    return (1/(b-a) * sympy.integrate(f, (t,a,b)))**(1/p)

if __name__ == "__main__":
    t = sympy.Symbol('x')
    tau = sympy.Symbol('\\tau')
    w = sympy.Symbol('w')
    p = sympy.Symbol('p')
    result = sympy.simplify(pdev('x', t, tau - w, tau + w,sympy.sympify('3')))
    print(sympy.latex(sympy.simplify(sympy.expand(result))))
