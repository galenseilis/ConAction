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
    lower = np.array(I)[:,0]
    upper = np.array(I)[:,1]
    scale = upper - lower
    scale = np.prod(scale)
    result = integrate.nquad(f, I, *args, **kwargs)
    result = result[0]
    result /= scale
    return result

def minkowski_deviation(f, I, p=2, *args, **kwargs):
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
    numerator = covariance(F, I, *args, **kwargs)
    p = Pool()
    poolf = lambda f: minkowski_deviation(f, I, p=len(F), *args, **kwargs)
    denominator = p.map(poolf, F)
    denominator = np.prod(denominator)
    return numerator / denominator

if __name__ == "__main__":
    F = [lambda x,y,z: x+y+z for i in range(11)]
    I = [(0,1)]*len(F)
    print(pearson_correlation(F[:3], I[:3]))
