"""
Use a SAT solver to find solutions to Waffle.
"""

from typing import List, Tuple, Dict, Iterable
from string import ascii_lowercase
from enum import Enum
from itertools import chain, product
from collections import Counter
from pysat.formula import CNF, IDPool
from pysat.solvers import Solver
from pysat.card import CardEnc, EncType
from .clue import CLUES, COLOR, BOARD, SQUARE, get_clues, print_clues, waffle_board
from .get_words import get_words
from .group import minimal_element, to_transpositions

PLACEMENT = Dict[SQUARE, str]

def check_solution(initial: PLACEMENT,
                   final: PLACEMENT,
                   perm: List[Tuple[SQUARE, SQUARE]]) -> List[
                       Tuple[SQUARE, str, str]]:
    """
    Input:
      initial, final: mapping of squares to letters.
      perm: a permutation of squares given in cycle form
    Output:
      Check whether the permutation applied to initial
      is equal to final.
    """
    current = initial.copy()
    for elt1, elt2 in perm:
            current[elt1], current[elt2] = current[elt2], current[elt1]
    return [(key, val, final[key]) for key, val in current.items()
            if final[key] != current[key]]

def letter_clues(clues: CLUES) -> Dict[str,Tuple[SQUARE,COLOR]]:
    """
    Redistribute the clues according to letter.
    """
    out = {}
    for square, (letter, color) in clues.items():
        if letter not in out:
            out[letter] = []
        out[letter].append((square, color))
    return out

class Waffle:
    """
    Solve the waffle game from only the initial clues.
    """

    def __init__(self,
                 size: int = 5,
                 wordlist: str = 'wordle',
                 encoding: str = 'totalizer',
                 verbose: int = 0):

        self._board = list(waffle_board(size))
        self._squares = set(chain(*self._board))
        self._internal = {square
            for square, count in Counter(chain(*self._board))
            if count == 1}
        self._wordlist = list(get_words(wordlist, wlen = size))
        self._verbose = verbose
        if self._verbose > 0:
            print(f"There are {len(self._wordlist)} words in the word list")
        self._encoding = getattr(EncType, encoding, EncType.totalizer)
        self._clues = {}
        self._upper = True
        self._allow_yellow = True
        
    def _basic(self, nocard: bool = False):
        """
        The basic model before clues.
        """
        # Each square has a letter
        for square in self._squares:
            choice = [self._pool.id(('s', square, _))
                for _ in ascii_lowercase]
            self._cnf.extend(CardEnc.equals(lits = choice,
                                        bound = 1,
                                        encoding = EncType.ladder,
                                        vpool = self._pool))
    
        # every place has a word
        for ind, place in enumerate(self._board):
            placed = []
            for word in self._wordlist:
                word_var = self._pool.id(('w', word, ind))
                placed.append(word_var)
                # positions: all word variables in this row/col
                self._cnf.extend([[- word_var,
                                self._pool.id(('s', square, letter))]
                                for square, letter in zip(place, word)])
            self._cnf.extend(CardEnc.equals(lits = placed,
                                        bound = 1,
                                        encoding = EncType.ladder,
                                        vpool = self._pool))
        for word in self._wordlist:
            # The word can occur at most once
            lits = [self._pool.id(('w', word, ind))
                for ind, _ in enumerate(self._board)]
            self._cnf.extend(CardEnc.atmost(lits = lits,
                                        bound = 1,
                                        encoding = EncType.ladder,
                                        vpool = self._pool))
    def _restrict_letters(self):
        """
        Add clause to restrict the multiset of letters.
        """
        # Since the letters are permuted, the multiset of letters
        # is fully specified.
        for letter, count in self.letter_counts.items():
            occupied = [self._pool.id(('s', square, letter))
                for square in self._squares]
            self._cnf.extend(CardEnc.equals(lits = occupied,
                                    bound = count,
                                    encoding = self._encoding,
                                    vpool = self._pool))
        # Now rule out all of the 0 occurences
        zeros = set(ascii_lowercase).difference(self.letter_counts.keys())
        self._cnf.extend([[-self._pool.id(('s', square, letter))]
                        for square, letter in product(self._squares, zeros)])

    def _add_clues(self, nocard: bool = False):
        """
        Process the clues.
        """
        if not nocard:
            self._restrict_letters()

        # Now process the color clues
        # Green is easiest
        # I think that yellow/black can be described as follows:
        # If a row/column, has a yellows and b blacks
        # then that indicates that the remaining squares (excluding green)
        # in that row/column have exactly a of that letter.
        # This takes care of the 'funky/dingo'.
        # This is subtle.  I think that the following is true
        # For each row/col and letter clue in that row
        # That letter appears there *at most* the number of yellows
        # for the letter
        # In particular, if there are no yellow, the letter can't
        # appear at all
        for place in self._board:
            # Get the clues for this place
            local_clues = letter_clues({square: val
                for square, val in self._clues.items()
                if square in place})
            for letter, values in local_clues.items():
                # Count up the color codes for this letter
                lower = 0
                upper = 0
                nongreen = False
                # The remaining squares influenced by clues for this leter
                eligible = set(place)
                for square, color in values:
                    eligible.remove(square)
                    svar = self._pool.id(('s', square, letter))
                    if color == COLOR.green:
                        # This square is correct
                        self._cnf.append([svar])
                    else:
                        # This square is incorrect
                        nongreen = True
                        self._cnf.append([-svar])
                        if color == COLOR.yellow:
                            upper += 1
                            if square in self._internal:
                                lower += 1
                if not nongreen:
                    continue
                rest = [self._pool.id(('s', square, letter))
                    for square in eligible]
                if upper == 0:
                    self._cnf.extend([[- _] for _ in rest])
                else:
                    # if only blacks, all eligible are not this letter
                    if self._allow_yellow: 
                        if lower > 0:
                            self._cnf.extend(CardEnc.atleast(lits = rest,
                                                            bound = lower,
                                                            encoding = self._encoding,
                                                            vpool = self._pool))
                        # yellows should be an upper bound
                        if self._upper:
                            self._cnf.extend(CardEnc.atmost(lits = rest,
                                                            bound = upper,
                                                            encoding = self._encoding,
                                                            vpool = self._pool))
                    
                        

    @property
    def letter_counts(self) -> Counter:
        """
        Get the letter counts
        """
        return Counter((_[0] for _ in self._clues.values()))

    @property
    def initial_placement(self) -> PLACEMENT:

        return {key: val[0] for key, val in self._clues.items()}

    def solve_words(self, clue_file: str,
                    nocard: bool = False,
                    upper: bool = True,
                    allow_yellow: bool = False,
                    solver_name: str = 'cd153') -> Iterable[
                        Tuple[Dict[SQUARE, str], Dict[int, str]]]:
        """
        Use SAT solving
        """
        self._clues = get_clues(clue_file, self._board)
        self._upper = upper
        self._allow_yellow = allow_yellow
        # print out clues
        if self._verbose > 0:
            print_clues(self._clues)
        self._cnf = CNF()
        self._pool = IDPool()
        self._basic()
        self._add_clues(nocard = nocard)
        solver = Solver(name = solver_name, bootstrap_with = self._cnf,
            use_timer = True)
        while True:
            status = solver.solve()
            if self._verbose > 0:
                print(f"time = {solver.time()}")
                print(f"Statistics: {solver.accum_stats()}")
            if status:
                positive =[self._pool.obj(_)
                    for _ in solver.get_model() if _ > 0]
                square_values = [_ for _ in positive
                    if _ is not None and _[0] == 's']
                word_values = [_ for _ in positive
                    if _ is not None and _[0] == 'w']
                yield ({_[1]: _[2] for _ in square_values},
                       {_[2]: _[1] for _ in word_values})
                # Now forbid that value

                solver.add_clause([-self._pool.id(_)
                                   for _ in square_values])
                solver.add_clause([-self._pool.id(_)
                                   for _ in word_values])
            else:
                break

    def solve_puzzle(self, clue_file: str,
                     minimal_opts: Dict | None = None,
                     solver_opts: Dict | None = None) -> Iterable[
                         Tuple[Dict[SQUARE, str], Dict[int, str]]]:
        """
        """
        if solver_opts is None:
            solver_opts = {}
        if minimal_opts is None:
            minimal_opts = {}
        for letter_placement, word_placement in self.solve_words(
            clue_file, **solver_opts):

            initial = self.initial_placement.copy()
            solution = minimal_element(initial, letter_placement,
                                       **minimal_opts)
            perm = to_transpositions(solution)
            yield word_placement, perm
            # We should check the solution
            result = check_solution(initial, letter_placement, perm)
            if not len(result) > 0:
                print(f"Solution check failed: {result}!")
            else:
                print("Solution checks!")
