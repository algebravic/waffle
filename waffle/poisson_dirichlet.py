"""
Use the Poisson-Dirichlet process to uniformly sample
permutations of length n with exactly k cycles.

Reference:
https://mathoverflow.net/questions/3330/how-can-i-generate-random-permutations-of-n-with-k-cycles-where-k-is-much-lar
"""
from typing import List
from itertools import chain
import numpy as np
from .group import canonical

def feller_representation(num: int, theta: float) -> List[int]:
    """
    Generate n-1 samples of bernoulli random variables
    with Prob(X[i] == 1) = theta/(i + theta), for i=1,...,n-1
    Make the 0th and nth both 1.
    """
    tst = theta/(theta + np.arange(num))
    places = np.arange(num)[(np.random.random(num) <= tst)]
    return np.concatenate([places, np.array([num])])

def chinese_restaurant(num: int, theta: float) -> List[int]:
    """
      Dubins and Pitman. We add elements to the partition one by one.
      Element 1 starts in a block on its own.
      Thereafter, when we add element r+1,
      suppose there are currently m blocks whose sizes are
      n_1, n_2, ..., n_m.
      Add element r+1 to block i with probability n_i/(r+theta),
      for 1 <= i <= m,
      and put element r+1 into a new block on its own
      with probability theta/(r+theta).

      numpy.random.choice(elements, number, p=list(probabilities))

      Note that when we consider element r+1,
      we must have sum_i n_i = r.
      Thus sum_i (n_i/(r+theta)) = r/(r+theta).
    
    """
    blocks = [[0]]
    for ind in range(1, num):
        probs = (np.array(list(map(len(blocks))))/(ind + theta)).tolist()
        where = np.random.choice(range(ind), 1,
            p = probs + [1-sum(probs)])
        if where < ind:
            blocks[where].append(ind)
        else:
            blocks.append([ind])
    return blocks
    
def optimal_value(num: int, kval: int,
                  epsilon: float = 1.0e-8,
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

def rejection_sample(num: int, kval: int,
                     verbose: int = 0) -> List[List[int]]:
    """
    A random permutation with exactly k cycles.
    """
    theta = optimal_value(num, kval, verbose = verbose)
    tries = 0
    while True:
        tries += 1
        breaks = feller_representation(num, kval)
        if len(breaks) == kval + 1:
            break
    if verbose > 0:
        print(f"Succeeded in {tries} tries")
    values = np.random.permutation(np.arange(num)).tolist()
    return canonical([values[_[0]: _[1]]
            for _ in zip(breaks[: -1], breaks[1: ])])
