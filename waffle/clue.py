"""
Read a CLUE file
The first 5 non blank lines are the letters

The next 5 non blank lines are the colors
"""
from typing import Dict, Tuple, Iterable, List
from enum import Enum
from itertools import chain, product

COLOR = Enum('Color', ['green', 'yellow', 'black'])
SQUARE = Tuple[int, int]
CLUES = Dict[SQUARE, Tuple[str, COLOR]]
ROWCOL = List[SQUARE]
BOARD = List[ROWCOL]
CLUE_DIR = 'data/puzzles'

def waffle_board(size: int = 5):
    """
    The Waffle board.
    """
    return list(chain(
        [[(row, col) for col in range(size)]
         for row in range(0, size, 2)],
        [[(row, col) for row in range(size)]
         for col in range(0, size, 2)]))


def get_rows(board: BOARD) -> Iterable[ROWCOL]:
    """
    """
    auxdict = {}
    for row, col in set(chain(*board)):
        if row not in auxdict:
            auxdict[row] = set()
        auxdict[row].add((row, col))
    for row in sorted(auxdict.keys()):
        yield list(sorted(auxdict[row]))

def get_clues(fname: str, board: BOARD) -> CLUES:
    """
    Read in the clue file and create the CLUES dict.
    """
    color_dict = {'g': COLOR.green, 'b': COLOR.black, 'y': COLOR.yellow}
    with open(f"{CLUE_DIR}/{fname}.txt", 'r') as myfile:
        lines = [_[:-1].replace(" ", "") for _ in myfile.readlines()]
        goodlines = [_.lower() for _ in lines if len(_) > 0]
        # There should be 10 lines
        if not all((_.isalpha() for _ in goodlines)):
            raise ValueError("Clues must be all alphabetic")
        squares = list(sorted(set(chain(*board))))
        # validate input
        rows = list(get_rows(board))
        nrows = len(rows)
        if not (2 * nrows == len(goodlines)
                and all((len(_[0]) == len(_[1])
                         for _ in zip(rows, goodlines[: nrows])))
                and all((len(_[0]) == len(_[1])
                         for _ in zip(rows, goodlines[nrows: ])))
                ):
            raise ValueError("Clues has wrong size")
        clue_dict = {}
        for place, letters, clues in zip(rows, goodlines[: nrows],
                                       goodlines[nrows: ]):
            for square, letter, clue in zip(place, letters, clues):
                if clue not in color_dict:
                    raise ValueError(f"clue = {clue} not valid")
                clue_dict[square] = (letter, color_dict[clue])
        return clue_dict

def print_clues(clues: CLUES):
    """
    Print the clues
    """
    squares = set(clues.keys())
    ncols = max((_[1] for _ in squares)) + 1
    nrows = max((_[0] for _ in squares)) + 1
    color_dict = {COLOR.green: 'G', COLOR.yellow: 'Y', COLOR.black: 'B'}
    for row in range(nrows):
        template = 2 * ncols * [' ']
        this_row = [_ for _ in clues.items() if _[0][0] == row]
        for square, (letter, color) in this_row:
            rplace = square[1]
            template[2 * rplace] = letter
            template[2 * rplace + 1] = color_dict[color]
        print(''.join(template))
