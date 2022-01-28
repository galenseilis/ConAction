import os
from timeit import timeit

from conaction import estimators
from conaction import naive
import matplotlib.pyplot as plt
import numpy as np

def make_correlation_matrix(size):
    '''
    Construct a random correlation matrix.

    Covariances are sampled from a uniform
    probability distribution over the interval
    [-1,1].
    '''
    R = np.zeros((size, size))
    scores = np.random.uniform(-1, 1, size=int((size ** 2 - size) / 2))
    R[np.triu_indices(size,1)] = scores
    R += R.T
    np.fill_diagonal(R, 1)
    return R

def make_normal_data(m, n):
    '''
    Constructs a dataset that
    follows a multivariate normal
    distribution with zero means,
    unit variances, and covariances
    uniformly sampled from the interval
    [-1, 1].
    '''
    means = np.zeros(n)
    cov = make_correlation_matrix(n)
    return np.random.multivariate_normal(mean=means, cov=cov, size=m)

START = 10
N = 10000
STEP = 10

est_funcs = [estimators.kendall_tau, estimators.grade_entropy]
naive_funcs = [naive.kendall_tau, naive.grade_entropy]
for f1, f2 in zip(est_funcs, naive_funcs):
    for arity in [2, 3, 5, 8]:
        est_times_1 = []
        est_times_2 = []
        est_times_max = []
        naive_times = []
        for sample_size in range(START, N+STEP, STEP):
            print(f1.__name__, arity, sample_size)
            data = make_normal_data(sample_size, arity)
            est_times_1.append(timeit('f1(data, n_jobs=1)', number=1, globals=globals()))
            est_times_2.append(timeit('f1(data, n_jobs=2)', number=1, globals=globals()))
            est_times_max.append(timeit('f1(data, n_jobs=os.cpu_count())', number=1, globals=globals()))
            naive_times.append(timeit('f2(data)', number=1, globals=globals()))

        plt.scatter(range(START, N+STEP, STEP), naive_times, label='naive.py', s=1, color='b')
        plt.scatter(range(START, N+STEP, STEP), est_times_1, label=f'estimators.py:{1} thread', s=1)
        plt.scatter(range(START, N+STEP, STEP), est_times_2, label=f'estimators.py:{2} threads', s=1)
        plt.scatter(range(START, N+STEP, STEP), est_times_max, label=f'estimators.py:{os.cpu_count()} threads', s=1)
        plt.legend()
        plt.xlabel('Sample Size')
        plt.ylabel('Runtime (s)')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'est_vs_naive_runtime_{f1.__name__}_{arity}_{N}.pdf')
        plt.close()

est_funcs = [estimators.pearson_correlation,
 estimators.reflective_correlation,
 estimators.multisemideviation,
 estimators.circular_correlation,
 estimators.signum_correlation,
 estimators.spearman_correlation,
 estimators.wang_zheng_correlation]

naive_funcs = [naive.pearson_correlation,
 naive.reflective_correlation,
 naive.multisemideviation,
 naive.circular_correlation,
 naive.signum_correlation,
 naive.spearman_correlation,
 naive.wang_zheng_correlation]

for f1, f2 in zip(est_funcs, naive_funcs):
    for arity in [2, 3, 5, 8]:
        est_times = []
        naive_times = []
        for sample_size in range(START, N+STEP, STEP):
            print(f1.__name__, arity, sample_size)
            data = make_normal_data(sample_size, arity)
            est_times.append(timeit('f1(data)', number=1, globals=globals()))
            data = data.tolist()
            naive_times.append(timeit('f2(data)', number=1, globals=globals()))

        plt.scatter(range(START, N+STEP, STEP), naive_times, label='naive.py', s=1)
        plt.scatter(range(START, N+STEP, STEP), est_times, label='estimators.py', s=1)
        plt.legend()
        plt.xlabel('Sample Size')
        plt.ylabel('Runtime (s)')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(f'est_vs_naive_runtime_{f1.__name__}_{arity}_{N}.pdf')
        plt.close()

f1 = estimators.minkowski_deviation
f2 = naive.minkowski_deviation
for arity in [1]:
    est_times = []
    naive_times = []
    for sample_size in range(START, N+STEP, STEP):
        print(f1.__name__, arity, sample_size)
        data = make_normal_data(sample_size, arity).flatten()
        est_times.append(timeit('f1(data)', number=1, globals=globals()))
        data = data.tolist()
        naive_times.append(timeit('f2(data)', number=1, globals=globals()))

    plt.scatter(range(START, N+STEP, STEP), naive_times, label='naive.py', s=1)
    plt.scatter(range(START, N+STEP, STEP), est_times, label='estimators.py', s=1)
    plt.legend()
    plt.xlabel('Sample Size')
    plt.ylabel('Runtime (s)')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'est_vs_naive_runtime_{f1.__name__}_{arity}_{N}.pdf')
    plt.close()

f1 = estimators.misiak_correlation
f2 = naive.misiak_correlation
for arity in [3, 4, 6, 9]:
    est_times = []
    naive_times = []
    sizes = []
    for sample_size in range(START, N+STEP, STEP):
        print(f1.__name__, arity, sample_size)
        x = make_normal_data(sample_size, 1).flatten()
        y = make_normal_data(sample_size, 1).flatten()
        data = make_normal_data(sample_size, arity-2)
        est_times.append(timeit('f1(x, y, data)', number=1, globals=globals()))
        try:
            naive_times.append(timeit('f2(x, y, data)', number=1, globals=globals()))
            sizes.append(sample_size)
        except ValueError as e:
            print(e)

    plt.scatter(sizes, naive_times, label='naive.py', s=1)
    plt.scatter(range(START, N+STEP, STEP), est_times, label='estimators.py', s=1)
    plt.legend()
    plt.xlabel('Sample Size')
    plt.ylabel('Runtime (s)')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'est_vs_naive_runtime_{f1.__name__}_{arity}_{N}.pdf')
    plt.close()


f1 = estimators.trencevski_malceski_correlation
f2 = naive.trencevski_malceski_correlation
for arity in [2, 4, 6, 8]:
    left_arity = int(np.floor(arity/2))
    right_arity = int(np.ceil(arity/2))
    est_times = []
    naive_times = []
    sizes = []
    for sample_size in range(START, N+STEP, STEP):
        print(f1.__name__, arity, sample_size)
        X = make_normal_data(sample_size, left_arity)
        Y = make_normal_data(sample_size, right_arity)
        est_times.append(timeit('f1(X, Y)', number=1, globals=globals()))
        try:
            naive_times.append(timeit('f2(X, Y)', number=1, globals=globals()))
            sizes.append(sample_size)
        except ValueError as e:
            print(e)

    plt.scatter(sizes, naive_times, label='naive.py', s=1)
    plt.scatter(range(START, N+STEP, STEP), est_times, label='estimators.py', s=1)
    plt.legend()
    plt.xlabel('Sample Size')
    plt.ylabel('Runtime (s)')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'est_vs_naive_runtime_{f1.__name__}_{arity}_{N}.pdf')
    plt.close()

est_funcs = [estimators.grade_entropy,
 estimators.kendall_tau,
 estimators.pearson_correlation,
 estimators.reflective_correlation,
 estimators.multisemideviation,
 estimators.circular_correlation,
 estimators.minkowski_deviation,
 estimators.signum_correlation,
 estimators.spearman_correlation,
 estimators.taylor_correlation,
 estimators.wang_zheng_correlation]
