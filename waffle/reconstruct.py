"""
Display a waffle state using matplotlib.
"""
from typing import Tuple, List, Set, Iterable, Dict, Set
from itertools import chain
from more_itertools import bucket
from pysat.solvers import Solver
from pysat.card import EncType, CardEnc
from pysat.formula import IDPool, CNF
from .clue import COLOR, SQUARE, CLUES, BOARD, PLACEMENT
from .group import check_compatible

SQUARES = List[SQUARE]

def is_contiguous(squares: Tuple[SQUARE, SQUARE]) -> bool:
    """
    Determine if the two squares are contiguous.
    """
    return (abs(squares[0][0] - squares[1][0])
            + abs(squares[0][1] - squares[1][1]) == 1)

def connected(squares: List[SQUARE]) -> bool:
    """
    Determine if a sequence of squares is connected.
    """
    return all(map(is_contiguous, zip(squares[: -1],
                                      squares[1: ])))

def square_to_board(squares: Set[SQUARE]) -> BOARD:
    """
    The rows/columns are contigous squares
    in order by first/last coordinate.

    This only works if all vertical/horizontal segments are contiguous
    """
    by_row = bucket(squares, key = lambda _: _[0])
    rows = (sorted(by_row[_], key = lambda _: _[1])
        for _ in sorted(list(by_row)))
    by_col = bucket(squares, key = lambda _: _[1])
    cols = (sorted(by_col[_], key = lambda _: _[0])
        for _ in sorted(list(by_col)))
    yield from (_ for _ in chain(rows, cols) if connected(_))

def yellow_black(current: PLACEMENT, solution: PLACEMENT) -> Iterable[Tuple[int, Set[SQUARE], int]]:
    """
    Determine black/yellow color for the non-green squares.
    """
    if not check_compatible(current, solution):
        raise ValueError("arguments not compatible")
    board = list(square_to_board(current))
    greens = set((square for square in current.keys()
              if current[square] == solution[square]))
    non_greens = set(current.keys()).difference(greens)

    constraints = []
    for idx, place in enumerate(board):
        # Find all letters in current
        consider = non_greens.intersection(place)
        cletters = set(current[square] for square in consider)
        for letter in cletters:
            occurs = [square for square in consider
                if current[square] == letter]
            other = consider.difference(occurs)
            count = len([square for square in other
                if solution[square] == letter])
            if count > 0:
                yield (idx, occurs, count)

def assign_colors(current: PLACEMENT,
                  solution: PLACEMENT,
                  solver_name: str = 'cd153',
                  verbose: int = 0) -> CLUES:
    """
    Calculate the list of squares that should be colored yellow/black.
    """
    yellows = set()
    greens = set((square for square in current.keys()
              if current[square] == solution[square]))
    non_greens = set(current.keys()).difference(greens)
    constraints = list(yellow_black(current, solution))
    # Now process the constraints via a SAT solver
    if verbose > 0:
        print(f"There are {len(constraints)} to be assigned.")
    if constraints:
        placement = {}
        pool = IDPool()
        cnf = CNF()
        for idx, squares, count in constraints:
            lits = [pool.id(('s', idx, _)) for _ in squares]
            for square in squares:
                if square not in placement:
                    placement[square] = []
                placement[square].append(pool.id(('s', idx, square)))
            if count >= len(squares):
                cnf.extend([[_] for _ in lits])
            else:
                cnf.extend(CardEnc.equals(lits = lits,
                                          bound = count,
                                          vpool = pool,
                                          encoding = EncType.totalizer))
        solver = Solver(name = solver_name, bootstrap_with = cnf)
        status = solver.solve()
        if not status:
            raise ValueError("Yellow/black not realizable!")
        model = solver.get_model()
        pos = [pool.obj(_) for _ in model if _ > 0]
        neg = [pool.obj(- _) for _ in model if _ < 0]
        yellows = set([_[2] for _ in pos if _ is not None])
        blacks = set([_[2] for _ in neg if _ is not None])
    else:
        yellows = set()
        blacks = non_greens
    other = non_greens.difference(yellows)
    blacks.update(other)
    the_clues = {}
    assignments = [(greens, COLOR.green),
        (blacks, COLOR.black),
        (yellows, COLOR.yellow)]
    for squares, color in assignments:
        the_clues.update({square: (current[square], color)
                          for square in squares})
    return the_clues
