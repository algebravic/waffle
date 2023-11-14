"""
Embed a message in a waffle board.
"""
from typing import Tuple, List, Dict
from itertools import chain
from sympy.combinatorics.permutations import Permutation
from .waffle_solve import Waffle
from .clue import PLACEMENT
from .sample import MetropolisBoard
from .reconstruct import assign_colors

def set_puzzle(waf: Waffle,
               words: List[str],
               perm: Permutation) -> Tuple[PLACEMENT, PLACEMENT]:
    """
    Input:
      waf: A waffle instance
      words: a list of words
      perm: a permutation
    Output:
      The final and initial waffle boards
    """
    # Enforce argument compatibility
    squares = list(sorted(set(chain(*waf._board))))
    if not (len(waf._board) == len(words)
            and all((len(_[0]) == len(_[1])
                     for _ in zip(waf._board, words)))
            and perm.size == len(squares)):
        raise ValueError("Incompatible arguments")

    solution = dict(chain(*(zip(_[0], _[1])
                            for _ in zip(waf._board, words))))
    back = {square: idx for idx, square in enumerate(squares)}
    inv = perm ** (-1)
    initial = {squares[inv(back[square])]: val for square, val
               in solution.items()}
    return solution, initial
