"""
Use the Poisson-Dirichlet process to uniformly sample
permutations of length n with exactly k cycles.

Reference:
https://mathoverflow.net/questions/3330/how-can-i-generate-random-permutations-of-n-with-k-cycles-where-k-is-much-lar
"""
from typing import List
from itertools import chain
import numpy as np

def feller_representation(num: int, theta: float) -> List[int]:
    """
    Generate n-1 samples of bernoulli random variables
    with Prob(X[i] == 1) = theta/(i + theta), for i=1,...,n-1
    Make the 0th and nth both 1.
    """
    tst = theta/(theta + np.arange(num))
    places = np.arange(num)[(np.random.random(num) <= tst)]
    return np.concatenate([places, np.array([num])])

def optimal_value(num: int, kval: int,
                  epsilon: float = 1.0e-5,
                  verbose: int = 0) -> float:
    """
    Find theta such that
    F(theta) = sum_{i=1}^n theta/(i-1+theta) = k.

    Note: F(theta) = n - sum_{i=1}^n (1 - theta/(i-1+theta)) =
    n - G(theta), where G(theta) = sum_{i=1}^n (i-1)/(i-1+theta).
    So G'(theta) = - sum_{i=1}^n (i-1)/(i-1+theta)^2

    So we want G(theta) = n-k

    F(theta) = k if and only if G(theta) = n - k
    So G(theta) - (n - k) = (n - F(theta)) - n + k = k - F(theta)
    """
    theta = kval / np.log(num)
    iteration = 0
    while True:
        iteration += 1
        gval = kval - (theta / (theta + np.arange(num))).sum()
        gderiv = - ((np.arange(1, num)/(theta + np.arange(1, num)) ** 2)).sum()
        if verbose > 0:
            print(f"iteration {iteration}, gval = {gval}, theta = {theta}")
        theta_new = theta - gval / gderiv
        if abs(theta - theta_new) <= epsilon:
            return theta
        theta = theta_new

def rejection_sample(num: int, kval: int) -> List[List[int]]:
    """
    A random permutation with exactly k cycles.
    """
    theta = optimal_value(num, kval)
    while True:
        breaks = feller_representation(num, kval)
        if len(breaks) == kval + 1:
            break
    values = np.random.permutation(np.arange(num)).tolist()
    return [values[_[0]: _[1]]
            for _ in zip(breaks[: -1], breaks[1: ])]
