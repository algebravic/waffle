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
from .clue import CLUES, COLOR, BOARD, SQUARE, get_clues, print_clues

def get_words(data: str = 'data/wordlist.txt'):
    """
    Get the word list.
    """
    with open(data, 'r') as myfile:
        yield from ((_[: -1].lower() for _ in myfile.readlines()))

def big_model(board: BOARD,
              wordlist: Iterable[str],
              data = 'data/wordlist.txt'):
    """
    Get the big model for Waffle.  We'll add clauses later
    corresponding to the clues.
    """
    words = list(wordlist)
    print(f"There are {len(words)} words in the word list")
    cnf = CNF()
    pool = IDPool()
    places = board
    squares = set(chain(*places))
    # Each square has a letter
    for square in squares:
        choice = [pool.id(('s', square, _)) for _ in ascii_lowercase]
        cnf.extend(CardEnc.equals(lits = choice,
                                  bound = 1,
                                  encoding = EncType.ladder,
                                  vpool = pool))
    
    # every place has a word
    for ind, place in enumerate(places):
        placed = []
        for word in words:
            word_var = pool.id(('w', word, ind))
            placed.append(word_var)
            # positions: all word variables in this row/col
            cnf.extend([[- word_var, pool.id(('s', square, letter))]
                        for square, letter in zip(place, word)])
        cnf.extend(CardEnc.equals(lits = placed,
                                  bound = 1,
                                  encoding = EncType.ladder,
                                  vpool = pool))
    for word in words:
        # The word can occur at most once
        cnf.extend(CardEnc.atmost(lits =
                                  [pool.id(('w', word, ind))
                                   for ind, _ in enumerate(places)],
                                  bound = 1,
                                  encoding = EncType.ladder,
                                  vpool = pool))
    return cnf, pool
        

def process_clues(board: BOARD,
                  clues: CLUES,
                  cnf: CNF,
                  pool: IDPool,
                  encoding: str = 'totalizer'):
    """
    Process the clues
    """
    places = board
    squares = set(chain(*places))
    counts = Counter([_[0] for _ in clues.values()])
    test = Counter(chain(*places))
    internal = [square for square, count in test if count == 1]

    encode = getattr(EncType, encoding, EncType.totalizer)
    # Since the letters are permuted, the multiset of letters
    # is fully specified.
    for letter, count in counts.items():
        occupied = [pool.id(('s', square, letter))
            for square in squares]
        cnf.extend(CardEnc.equals(lits = occupied,
                                  bound = count,
                                  encoding = encode,
                                   vpool = pool))
    # Now rule out all of the 0 occurences
    zeros = set(ascii_lowercase).difference(counts.keys())
    cnf.extend([[-pool.id(('s', square, letter))]
                 for square, letter in product(squares, zeros)])
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
    for place in places:
        # Get the clues for this place
        local_clues = [(square, clues[square]) for square in place]
        # First place the green, if any
        letter_dict = {}
        eligible = set() # Non green squares in this row/col
        present = set()
        for square, (letter, clue) in local_clues:
            if clue == COLOR.green:
                cnf.append([pool.id(('s', square, letter))])
            else:
                cnf.append([-pool.id(('s', square, letter))])
                present.add(letter)
                eligible.add(square)
                if letter not in letter_dict:
                    letter_dict[letter] = []
                letter_dict[letter].append((square, clue))
        # forbid all letters not mentioned in nongreen squares
        # to_remove = set(ascii_lowercase).difference(present)
        # cnf.extend([[-pool.id(('s', square, letter))]
        #             for square, letter in product(eligible, to_remove)]
        #           )
        # yellow/black can't be in the indicated squares
        for letter, clue_pos in letter_dict.items():
            yellows = 0
            # Will be squares in this row/col not green and not
            # mentioned by this letter.
            toplace = eligible.copy()
            # Yellow or black can't be in this square
            for square, clue in clue_pos:
                cnf.append([-pool.id(('s', square, letter))])
                if clue == COLOR.yellow:
                    yellows += 1
                    if square in internal:
                        # It occurs somewhere in this row/col
                        cnf.append([pool.id(('s', _, letter))
                                    for _ in place])
                elif clue != COLOR.black:
                    raise ValueError(f"Illegal color {clue}")
                toplace.remove(square)
            # treat zeros specially
            # if yellows == 0:
            #     cnf.extend([[-pool.id(('s', square, letter))]
            #                  for square in toplace])
            # else:
            #     pass
                # lits = [pool.id(('s', _, letter)) for _ in toplace]
                # cnf.extend(CardEnc.atmost(
                #     lits = lits,
                #     bound = yellows,
                #     encoding = encode,
                #     vpool = pool))
        
def solve_waffle(board: BOARD,
                 clue_file: str,
                 wordlist: Iterable[str],
                 encoding: str = 'totalizer',
                 solver_name: str = 'cd153') -> Iterable[Dict[SQUARE, str]]:
    """
    Use SAT solving
    """
    cnf, pool = big_model(board, wordlist)
    clues = get_clues(clue_file, board)
    # print out clues
    print_clues(clues)
    process_clues(board, clues, cnf, pool, encoding = encoding)
    solver = Solver(name = solver_name, bootstrap_with = cnf)
    while True:
        status = solver.solve()
        if status:
            positive =[pool.obj(_) for _ in solver.get_model() if _ > 0]
            square_values = [_[1:] for _ in positive
                if _ is not None and _[0] == 's']
            word_values = [_ for _ in positive if _ is not None
                           and _[0] == 'w']
            yield dict(square_values), word_values
            # Now forbid that value
            solver.add_clause([-pool.id(_) for _ in word_values])
        else:
            break
