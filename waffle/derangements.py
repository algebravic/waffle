"""
Routines about derangements and permutations.
"""
from typing import List, Tuple
from functools import cache, partial
from random import randint
from itertools import chain
from sympy import factorial

CYCLE = List[int] | Tuple[int]
CYCLE_PERM = List[CYCLE] | Tuple[CYCLE]

@cache
def associated(num: int, kval: int) -> int:
    if num <= 2 * kval - 1:
        return 0
    elif kval == 1:
        return int(factorial(num - 1))
    elif num > 1:
        return (num - 1) * (associated(num - 1, kval)
                            + associated(num - 2, kval - 1))
    else:
        return 0

def all_distinct(elt: CYCLE_PERM) -> bool:
    """
    Check that all elements are distinct.
    """
    supp = list(chain(*elt))
    return len(supp) == len(set(supp))

def _validate_cycle(cperm: CYCLE_PERM) -> bool:
    """
    Test if a cycle permutation is valid
    0) it is a list of list of integers.
    1) It consists of distinct non-negative integers
    """
    return (isinstance(cperm, (list, tuple))
            and all((isinstance(elt, (list, tuple))
                     and len(elt) > 0
                     and all((isinstance(_, int)
                              and _ >= 0
                              for _ in elt)
                             )
                     for elt in cperm))
            and all_distinct(cperm))

def _rotate(cycle: CYCLE) -> CYCLE:
    """
    Rotate a cycle so that its maximum element
    is at the beginning.
    """
    max_elt = max(cycle)
    idx = cycle.index(max_elt)
    return cycle[idx : ] + cycle[: idx]

def standard_form(cperm: CYCLE_PERM) -> CYCLE_PERM:
    """
    Put a cycle permutation in standard form.
    """
    if not _validate_cycle(cperm):
        raise ValueError("cperm is not a list|tuple of list|tuple of int")
    return tuple(sorted(tuple(map(_rotate, cperm)),
                        key = lambda _: _[0]))

def _promote(val: int, arg: Tuple[int]) -> Tuple[int]:
    return tuple((_ + (_ >= val) for _ in arg))

def _demote(val: int, arg: Tuple[int]) -> Tuple[int]:
    return tuple((_ - (_ > val) for _ in arg))

def _insert(val: int, ins: int, cycle: Tuple[int]) -> Tuple[int]:
    """
    Insert ins in the cycle after val, if it is present.
    """
    if val not in cycle:
        return cycle
    idx = cycle.index(val)
    return  (ins,) + cycle[idx + 1: ] + cycle[: idx + 1]

def unrank_associated(num: int, kval: int, idx: int) -> CYCLE_PERM | None:
    """
    Produce the permutation in cycle form given by idx
    for degree n and k cycles.
    """
    # First check for validity
    if not (idx >= 0 and idx < associated(num, kval)):
        raise ValueError(f"Index out of bounds ({num},{kval},{idx})")

    if num == 2 and kval == 1:
        return ((0,1),)
    elif num > 2:
        val1 = associated(num - 2, kval - 1)
        val2 = associated(num - 1, kval)
        val = val1 + val2
        if val == 0:
            return None
        quo = idx // val
        rem = idx % val
        if rem < val2:
            sub_perm = unrank_associated(num - 1, kval, rem)
            return tuple(map(partial(_insert, quo, num - 1), sub_perm))
        else:
            rem_prime = rem - val2
            sub_perm = unrank_associated(num - 2, kval - 1, rem_prime)
            return tuple(map(partial(_promote, quo), sub_perm)) + ((quo, num - 1),)

    else:
        return None

def rank_associated(cperm: CYCLE_PERM) -> int:
    """
    calculate the rank associate with this permutation
    """
    if not _validate_cycle(cperm):
        raise ValueError(f"{cperm} is not a valid cycle permutation")
    big = max(chain(*cperm))
    num = big + 1
    if num == 2:
        return 0
    kval = len(cperm)
    val1 = associated(num - 2, kval -1)
    val2 = associated(num - 1, kval)
    val = val1 + val2
    idx, target = [_ for _ in enumerate(cperm) if big in _[1]][0]
    the_cycle = _rotate(target) # largest is at the beginning
    multiplier = the_cycle[-1] # insertion point is at the end
    addend = multiplier * val
    if len(the_cycle) > 2:
        # Remove n from its containing cycle
        sub_perm = cperm[: idx] + (the_cycle[1:],) + cperm[idx + 1:]
        return addend + rank_associated(sub_perm)
    elif len(the_cycle) == 2: # len of cycle = 2
        # Remove both n and its partner from the permutation
        sub_perm = tuple(map(partial(_demote, multiplier),
            cperm[: idx] + cperm[idx + 1:]))
        return addend + val2 + rank_associated(sub_perm)
    else:
        raise ValueError("Cycle of size 1 encountered")

def test_rank(num: int, kval: int) -> bool:
    """
    Generate a random associated permutation of type (n,k).
    First choose a uniform random in [0,d(n,k))
    and call unrank.  Then rank the resulting permutation
    and test if the resulting index is equal to the random int.
    """
    idx = randint(0, associated(num, kval) - 1)
    cperm = unrank_associated(num, kval, idx)
    rnk = rank_associated(cperm)
    return idx == rnk
