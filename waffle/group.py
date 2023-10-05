"""
Calculations of group structure.
"""
from typing import Tuple, Iterable, List, Set, Dict, Hashable
from itertools import product, chain, tee
from collections import Counter
from sympy.combinatorics.permutations import Permutation 
from sympy.combinatorics.perm_groups import PermutationGroup
from .clue import SQUARE

PLACEMENT = Dict[SQUARE, str]
REVERSE = Dict[str, List[SQUARE]]
SQUARE_PERM = List[List[SQUARE]]

def _validate(placement: PLACEMENT):

    if not (isinstance(placement, dict)
            and all((isinstance(_, tuple)
                     and len(_) == 2
                     and all((isinstance(val, int) for val in _))
                     for _ in placement.keys()))
            and all((isinstance(_, str) for _ in placement.values()))):
        raise ValueError("Input must be a dict indexed by pairs of ints with string values")

def _reverse(placement: PLACEMENT) -> REVERSE:
    """
    Reverse a dictionary
    """
    out = {}
    for key, val in placement.items():
        if val not in out:
            out[val] = set()
        out[val].add(key)
    return out

def initial_permutation(initial: PLACEMENT, solution: PLACEMENT) -> Permutation:
    """
    Inputs: intial and solution are dictionaries with the same key set.
    and value multiset.  Output a permutation of the key set,
    which transforms initial to solution.
    """

    # Check input
    _validate(initial)
    _validate(solution)
    if not (set(initial.keys()) == set(solution.keys())
            and Counter(initial.values()) == Counter(solution.values())):
        raise ValueError("initial and placement not permutable")
    outperm = {}
    rinit = _reverse(initial)
    rsoln = _reverse(solution)

    for key, val in rinit.items():

        outperm.update(zip(sorted(val), sorted(rsoln[key])))
    return outperm

def placement_partition(placement: PLACEMENT) -> List[Set[SQUARE]]:

    _validate(placement)
    return list(_reverse(placement).values())

def sym_gens(elts: List[int]) -> Iterable[SQUARE]:
    """
    Coxeter generators of the full symmetric group.
    """
    yield from zip(elts[: -1], elts[1: ])


def to_cycle(perm: Dict[Hashable, Hashable]) -> List[List[Hashable]]:
    """
    Cycle rendition of a permutation.
    """
    consider = set(perm.keys())
    used = set()
    cycles = []
    while consider:
        first = consider.pop()
        cycle = [first]
        elt = perm[first]
        while elt != first:
            cycle.append(elt)
            consider.remove(elt)
            elt = perm[elt]
        if len(cycle) > 1:
            cycles.append(cycle)
    return cycles

def climb(perm: Permutation, grp: PermutationGroup,
          tenure: int = 10) -> Tuple[Permutation, int]:
    """
    Do one best ascent climb until a hilltop.
    """
    tst = perm
    tabu = []
    tries = 0
    while True:
        tries += 1
        candidates = [_ * tst for _ in grp.generators]
        consider = [_ for _ in candidates if _ not in tabu]
        nbr = max(consider, key = lambda _: _.cycles)
        # Have we reached a hilltop?
        if nbr.cycles < tst.cycles:
            break
        tst = nbr
        tabu.append(tst)
        if len(tabu) > tenure:
            tabu = tabu[1:]
    return tst, tries

def hillclimb(perm: Permutation, grp: PermutationGroup,
              tenure: int = 10,
              limit: int | None = None,
              iterations: int = 100,
              trace: int = 0) -> Permutation:
    """
    Do a discrete hillclimb to find (or approximate) a permutation
    in the coset h G, which has the largest number of cycles.

    Do best ascent.  When reaching a hilltop, make a random restart.
    """
    best = perm
    count = Counter()
    for iteration in range(1, iterations + 1):
        start = grp.random() * perm
        tst, tries = climb(start, grp, tenure = tenure)
        count.update([tries])
        if tst.cycles > best.cycles:
            best = tst
            if limit is not None and best.cycles >= limit:
                break
        if trace > 0 and iteration % trace == 0:
            print(f"Iteration {iteration}")
            print(f"Best = {best.cycles}")
            print(f"tst = {tst.cycles}")
        # Keep going until we reach a hilltop
    print(f"histogram = {count}")
    return best

def exhaust_coset(coset_rep: Permutation, subgrp: PermutationGroup,
                  limit: int | None = None) -> Permutation:
    """
    Input:
      coset_rep: a representative of a right coset, h
      subgrp: A subgroup, G
    Exhaust the right coset G h to find the an element
    with the largest number of cycles, or if limit is not None
    an element with at least limit cycles.
    """
    return max((_ * coset_rep for _ in subgrp._elements),
               key = lambda _: _.cycles)

# sympy permutations need integers
def minimal_element(initial: PLACEMENT, solution: PLACEMENT,
                    exhaust: bool = False,
                    hillclimb_opts: Dict | None = None) -> SQUARE_PERM:

    # First create the mapping to a from indices
    forward = dict(enumerate(sorted(initial.keys())))
    back = {_[1]: _[0] for _ in forward.items()}
    degree = len(initial.keys())
    print(f"degree = {degree}")

    iperm = initial_permutation(initial, solution)
    tperm = Permutation([back[_[1]] for _ in sorted(iperm.items())])
    # Find the invariant subgroup
    part = placement_partition(solution)
    parts = map(lambda _: [back[elt] for elt in _], part)
    # Create the Coxeter generators
    gens = chain(*map(sym_gens, parts))
    grp = PermutationGroup([Permutation([_], size = degree)
        for _ in gens])
    print(f"Group size = {grp.order()}")
    
    if exhaust:
        max_elt = exhaust_coset(tperm, grp)
    else: # Do a hillclimb
        if hillclimb_opts is None:
            hillclimb_opts = {}
        max_elt = hillclimb(tperm, grp, **hillclimb_opts)
    # Now convert back to cyclic form
    return [[forward[_] for _ in cycle]
            for cycle in max_elt.cyclic_form]

def _validate_cycle(permutation: List[List[Hashable]]):
    """
    Validate cyclic form
    """
    support = list(chain(*permutation))
    if (len(support) != len(set(support))
        or any(map(lambda _: len(_) == 0, permutation))):
        raise ValueError("Elements are not distinct!")

def cycle_to_dict(permutation: List[List[Hashable]]) -> Dict[
    Hashable, Hashable]:
    """
    Translate from cyclic form to a dict
    """
    _validate_cycle(permutation)
    out = {}
    # Elements should be distinct

    return dict(chain(*(zip(cycle, cycle[1: ] + [cycle[0]])
                        for cycle in permutation)))

def to_transpositions(permutation: List[List[Hashable]]) -> List[
    Tuple[Hashable, Hashable]]:
    """
    Convert cyclic form into a product of transpositions.
    """
    _validate_cycle(permutation)
    return list(reversed(list(chain(*(list(zip(cycle[: - 1], cycle[1: ]))
                    for cycle in permutation)))))