"""
Permutations on n letters having k cycles.
"""
from typing import List, Tuple
from functools import cache, partial
from random import randint
from itertools import chain
from sympy import factorial
from .derangements import CYCLE, CYCLE_PERM
from .derangements import _rotate, _validate_cycle, _insert, _demote

@cache
def stirling(num: int, kval: int) -> int:
    """
    Stirling number of the first kind

    s(n,k) = s(n-1,k-1) + (n-1) s(n-1,k)
    Either n is in a cycle by itself (first term), where
    1..n-1 form a k-1 cycle permutation
    or n is adjoined to a cycle in a k-cycle permutation
    of n-1 elements.  It can be place in n-1 ways:
    after each element in 1..n-1 in its cycle.
    """
    if kval == 1:
        return int(factorial(num - 1))
    elif num > 1:
        return (stirling(num - 1, kval - 1)
                + (num - 1) * stirling(num - 1, kval))
    else:
        return 0

def unrank_stirling(num: int, kval: int, idx: int) -> CYCLE_PERM | None:
    """
    Unrank a permutation.
    """
    if not (idx >= 0 and idx < stirling(num, kval)):
        raise ValueError(f"Index out of bounds ({num}, {kval}, {idx})")

    if num == 1 and kval == 1:
        return ((0,),)
    elif num > 1:
        val1 = stirling(num - 1, kval - 1)
        val2 = stirling(num - 1, kval)

        if idx < val1:
            sub_perm = unrank_stirling(num - 1, kval - 1, idx)
            return sub_perm + ((num - 1, ),)
        else:
            rest = idx - val1
            if val2 == 0:
                return None
            quo = rest // val2
            rem = rest % val2
            sub_perm = unrank_stirling(num - 1, kval, rem)
            return tuple(map(partial(_insert, quo, num - 1), sub_perm))

def rank_stirling(cperm: CYCLE_PERM) -> int:
    """
    Calculate the stirling rank of the permutation.
    """
    if not _validate_cycle(cperm):
        raise ValueError(f"{cperm} is not a valid cycle permutation")
    big = max(chain(*cperm))
    num = big + 1
    if num == 1:
        return 0
    idx, target = [_ for _ in enumerate(cperm) if big in _[1]][0]
    the_cycle = _rotate(target) # largest is at the beginning
    if len(the_cycle) == 1:
        sub_perm = cperm[: idx] + cperm[idx + 1: ] # remove the cycle
        return rank_stirling(sub_perm)
    else:
        kval = len(cperm)
        val1 = stirling(num - 1, kval - 1)
        val2 = stirling(num - 1, kval)
        multiplier = the_cycle[-1] # insertion point is at the end
        # remove n from its cycle
        sub_perm = cperm[: idx] + (the_cycle[1: ],) + cperm[idx + 1: ]
        return val1 + multiplier * val2 + rank_stirling(sub_perm)

def test_stirling_rank(num: int, kval: int) -> bool:
    """
    Generate a random stirling permutation of type (n,k).
    First choose a uniform random in [0,d(n,k))
    and call unrank.  Then rank the resulting permutation
    and test if the resulting index is equal to the random int.
    """
    idx = randint(0, stirling(num, kval) - 1)
    cperm = unrank_stirling(num, kval, idx)
    rnk = rank_stirling(cperm)
    return idx == rnk
