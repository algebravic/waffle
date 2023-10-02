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
from .get_words import system_words

def get_words(data: str = 'data/wordlist.txt'):
    """
    Get the word list.
    """
    with open(data, 'r') as myfile:
        yield from ((_[: -1].lower() for _ in myfile.readlines()
                    if _[: - 1].isalpha()))

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
        if wordlist == 'system':
            self._wordlist = list(system_words(size))
        else:
            self._wordlist = list(get_words(f'data/{wordlist}.txt'))
        self._verbose = verbose
        if self._verbose > 0:
            print(f"There are {len(self._wordlist)} words in the word list")
        self._encoding = getattr(EncType, encoding, EncType.totalizer)
        self._internal = set([square
            for square, count in Counter(chain(*self._board))
            if count == 1])
        self._clues = {}
        
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
            local_clues = [(square, self._clues[square]) for square in place]
            # First place the green, if any
            letter_dict = {}
            eligible = set() # Non green squares in this row/col
            for square, (letter, clue) in local_clues:
                if clue == COLOR.green:
                    # If green fix the value of letter in square
                    self._cnf.append([self._pool.id(('s', square, letter))])
                else:
                    # if not green, letter can't be on this square
                    self._cnf.append([-self._pool.id(('s', square, letter))])
                    eligible.add(square)
                    if letter not in letter_dict:
                        letter_dict[letter] = []
                    letter_dict[letter].append((square, clue))
            # yellow/black can't be in the indicated squares
            for letter, clue_pos in letter_dict.items():
                yellow_internal = 0
                yellow_external = 0
                # Will be squares in this row/col not green and not
                # mentioned by this letter.
                toplace = eligible.copy()
                # Yellow or black can't be in this square
                for square, clue in clue_pos:
                    if clue == COLOR.yellow:
                        if square in self._internal:
                            yellow_internal += 1
                        else:
                            yellow_external += 1
                    elif clue != COLOR.black:
                        raise ValueError(f"Illegal color {clue}")
                    toplace.remove(square)
                # treat zeros specially
                lits = [self._pool.id(('s', _, letter)) for _ in toplace]
                # If there are no yellows, then this letter can't
                # occur in this row/column.
                if yellow_internal + yellow_external == 0:
                    self._cnf.extend([[- _] for _ in lits])

                else:
                    # Below doesn't seem to work.
                    # cnf.extend(CardEnc.atmost(
                    #     lits = lits,
                    #     bound = yellow_internal + yellow_external,
                    #     encoding = encode,
                    #     vpool = self._pool))
                    pass
                if yellow_internal > 0:
                    self._cnf.extend(CardEnc.atleast(
                        lits = lits,
                        bound = yellow_internal,
                        encoding = encode,
                        vpool = self._pool))

    @property
    def letter_counts(self) -> Counter:
        """
        Get the letter counts
        """
        return Counter((_[0] for _ in self._clues.values()))

    @property
    def initial_placement(self):

        return {key: val[0] for key, val in self._clues.items()}

    def solve(self, clue_file: str,
              nocard: bool = False,
              solver_name: str = 'cd153') -> Iterable[Dict[SQUARE, str]]:
        """
        Use SAT solving
        """
        self._clues = get_clues(clue_file, self._board)
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
                positive =[self._pool.obj(_) for _ in solver.get_model() if _ > 0]
                square_values = [_ for _ in positive
                    if _ is not None and _[0] == 's']
                word_values = [_ for _ in positive if _ is not None
                    and _[0] == 'w']
                yield ({_[1]: _[2] for _ in square_values},
                       {_[2]: _[1] for _ in word_values})
                # Now forbid that value
                # solver.add_clause([-self._pool.id(_) for _ in word_values])
                solver.add_clause([-self._pool.id(_) for _ in square_values])
            else:
                break
