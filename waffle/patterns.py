"""
Test Clues for left/right symmetry.
"""
from typing import List
from .clue import CLUES, BOARD, SQUARE, get_clues, waffle_board
from .reconstruct import square_to_board

def is_horizontal(place: List[SQUARE]) -> bool:
    """
    Test if a place is horizontal
    """
    return place[0][0] == place[1][0]

def is_vertical(place: List[SQUARE]) -> bool:
    """
    Test if a place is horizontal
    """
    return place[0][1] == place[1][1]

def is_symmetric(place: List[SQUARE], clues: CLUES) -> bool:
    """
    Test if the clues in a place are symmetric
    """
    colors = [clues[_][1] for _ in place]
    return all((_[0] == _[1] for _ in zip(colors, reversed(colors))))

def sym_test(clues: CLUES) -> bool:
    """
    Test for left/right symmetry.
    """
    horizontal = [place for place in square_to_board(clues.keys())
                  if is_horizontal(place)]
    vertical = [place for place in square_to_board(clues.keys())
                  if is_vertical(place)]
    return (all((is_symmetric(place, clues) for place in horizontal)),
            all((is_symmetric(place, clues) for place in vertical)))

def test_sym(name: str, size: int = 5) -> bool:
    """
    Get a clue file and test it.
    """
    return sym_test(get_clues(name, waffle_board(size)))
