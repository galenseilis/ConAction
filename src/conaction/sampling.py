import numpy as np

def permute_columns(x):
    '''
    Permutes the columns of a data matrix.
    '''
    ix_i = np.random.sample(x.shape).argsort(axis=0)
    ix_j = np.tile(np.arange(x.shape[1]), (x.shape[0], 1))
    return x[ix_i, ix_j]

