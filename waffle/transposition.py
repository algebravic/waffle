"""
Count permutations of n letters with k cycles
which can be obtained as a product of n-k transpositions.
"""
from typing import List, Tuple, Set, Dict, Iterable, Any
from functools import partial
from itertools import combinations
import networkx as nx
from sympy.utilities.iterables import partitions

PART = Tuple[Tuple[int, int], ...]
DPART = Dict[int, int]
FUNC = Callable[[Any], Any]

def _normalize(dct: Dict[int, int]) -> Dict[int, int]:
    """
    Remove 0 weight entries
    """
    return tuple(sorted([_ for _ in dct.items()
                        if _[1] > 0]))

def _total(dct: Dict[int, int]) -> int:
    """
    Total weight of a partition.
    """
    return sum((_[0] * _[1] for _ in dct.items()))

def _check(standard: PART, nxt: DPART, msg: str):
    """
    
    """
    if _total(nxt) != _total(dict(standard)):
        print(f"Error {msg}: {standard} -> {_normalize(nxt)}")

def function_power(func: FUNC, power: int) -> FUNC:
    """
    Function power.
    """
    if power < 0:
        raise ValueError(f"power = {power} not allowed.")
    elif power == 0:
        return lambda _: _
    elif power == 1:
        return func
    else:
        return lambda _: function_power(func, power - 1)(func(_))

def transposition_edges(num: int) -> Iterable[Tuple[DPART, DPART, int]]:
    """
    Produce a digraph whose nodes are the partitions
    of n, and whose weighted edges correspond to
    multiplying by the conjugacy class of a transposition.
    More specifically if K_1 denotes the conjugacy class
    of transpositions, and $K_t$ is any other conjugacy
    class, the *connection coefficients* are c(t,u)
    where K_1 K_t = sum_u c(t,u) K_u.  We can calculte
    them as follows:

    Multiplying by a transposition will either merge
    two cycles if the two elements of the transposition
    are each in one of the cycles.  If the two elements
    are both in the same cycle, then it will split
    that cycle in two.  Specifically, if the containing
    cycle is length m, and the cyclic distance
    between the positions of the two elements is d,
    it will split the cycle into two cycles of length
    d and m-d. In the first case, if the cycles are
    of lengths m1 and m2 respectively, there are
    exactly m1 * m2 transpositions.  In the second
    case, it the cycle is length m, we have pairs
    (i, i+d), with 1 <= i, i+d <=m,which give
    the pair of cycles of length d and m-d.
    So there are m-d such pairs.  We need to take
    multplicity into account.

    In the first case, if the multiplicities
    are k1 and k2 respectively, we have
    m1**k1 * m2 ** k2.  In the second case, if
    the multipicity of the m cycle is k, we raise
    things to the k-th power, so that are (m-d) ** k
    pairs that produce a splitting into (d,m-d).  In
    all cases we must have 1 <= d <= m-1.
    """
    for part in partitions(num):
        standard = _normalize(part)
        total_weight = 0
        for (siz1, mult1), (siz2, mult2) in combinations(standard, 2):
            reduced = part.copy()
            # join two parts of sizes siz1, siz2 (unequal)
            reduced[siz1] -= 1
            reduced[siz2] -= 1
            reduced[siz1 + siz2] = reduced.get(siz1 + siz2, 0) + 1
            weight = siz1 * mult1 * siz2 * mult2
            total_weight += weight
            _check(standard, reduced, 'join')
            yield (standard, reduced, weight)
        # single cycle
        for siz, mult in standard:
            if mult > 1: # We can combine two of the same size
                reduced = part.copy()
                reduced[siz] -= 2
                reduced[2 * siz] = reduced.get(2 * siz, 0) + 1
                weight = (((mult * (mult - 1)) // 2) * siz ** 2)
                total_weight += weight
                _check(standard, reduced, 'inner join')
                yield (standard, reduced, weight)
            # split a single part
            reduced = part.copy()
            reduced[siz] -= 1
            # (a,b), 1 <= a < b <= size
            # 1 <= start, start + delta <= size
            # size - delta >
            accum = {}
            for start, stop in combinations(range(siz), 2):
                delta = stop - start
                key = tuple(sorted((delta, siz - delta)))
                accum[key] = accum.get(key, 0) + 1
            for (siz1, siz2), sub_weight in accum.items():
                cred = reduced.copy()
                cred[siz1] = cred.get(siz1, 0) + 1
                cred[siz2] = cred.get(siz2, 0) + 1
                weight = sub_weight * mult
                total_weight += weight
                _check(standard, cred, f'split ({siz1},{siz2})')
                yield (standard, cred, weight)
        if total_weight != (num * (num - 1)) // 2:
            print(f"{standard} weight = {total_weight}")
                
def transposition_graph(num: int) -> nx.DiGraph:
    """
    Produce the directed graph.
    """
    gph = nx.DiGraph()
    for source, sink, weight in transposition_edges(num):
        gph.add_edge(source, _normalize(sink), weight = weight)
    return gph

def transition(gph: nx.DiGraph,
               state: Dict[DPART, int]) -> Dict[DPART, int]:
    """
    From a weighted state: a weighted vertex set
    and a digraph: follow the edges.
    """

    res = {}
    for vtx, wgt in state.items():
        for source, sink in gph.edges(vtx):
            add_wgt = gph.edges[source, sink]['weight'] * wgt
            res[sink] = res.get(sink, 0) + add_wgt
    return res

def power(gph: nx.DiGraph,
          mult: int,
          state: Dict[DPART, int]) -> Dict[DPART, int]:

    if mult <= 0:
        return state
    tmp = state
    for ind in range(mult):
        tmp = transition(gph, tmp)
    return tmp

def by_cycles(state: Dict[DPART, int]) -> Dict[int, int]:
    """
    Accumulate weight by number of cycles.
    """
    res = {}
    for part, wgt in state.items():
        cycles = sum((_[1] for _ in part))
        res[cycles] = res.get(cycles, 0) + wgt
    return res

def cycle_census(num: int, kval: int) -> Dict[int, int]:
    """
    Return the census by number of cycles for
    independently multiplying k transpositions together,
    uniformly chosen in permutations of degree n.
    """
    # start: concentrated on the identity
    return by_cycles(power(transposition_graph(num),
                           kval,
                           {((1, num),) : 1}))
