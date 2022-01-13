import functools
from operator import mul

def pearson_correlation(x):
    '''
    Naive implementation of multilinear
    pearson correlation coefficient.

    Parameters
    ------
    x : 2d list
        Data table (columns index variables, rows index samples)

    Returns
    ------
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

if __name__ == '__main__':
    import numpy as np
    import estimators
    from timeit import timeit
    x = np.random.random(100000).reshape((10000,10))
    x = [[float(j) for j in i] for i in x]
    print(timeit('pearson_correlation(x)', number=1, globals={'x':x, 'pearson_correlation': pearson_correlation}))
    x = np.array(x, dtype=np.float64)
    print(timeit('pearson_correlation(x)', number=1, globals={'x':x, 'pearson_correlation': estimators.pearson_correlation}))
