"""
Construct a random Waffle Game
"""
from typing import List, Tuple, Dict, Set, Iterable
from random import choice, choices, randint, random
from math import exp, log
from string import ascii_lowercase
from itertools import chain
from more_itertools import bucket
from scipy.sparse import csr_matrix
from sklearn.neighbors import KDTree
from .clue import BOARD, SQUARE

def wordlist_to_vec(wordlist: List[str]):
    """
    Convert a wordlist to a 1-hot sparse encoding.
    """
    degree = len(wordlist[0])
    num = len(wordlist)
    mat = csr_matrix((degree * num * [1.0],
                      (list(chain(*(degree * [_]
                                    for _ in range(num)))),
                       list(chain(*[[ord(word[_]) - ord('a')
                                     + 26 * _
                                    for _ in range(degree)]
                                    for word in wordlist])))),
                      shape = (num, 26 * degree))
    return mat

def print_board(board: BOARD, answer: Dict[SQUARE, str]) -> Iterable[str]:
    """
    Format the answer 
    """
    for rowcol in board:
        yield ''.join((answer[_] for _ in rowcol))

def _word_buckets(wordlist: List[str]) -> Dict[Tuple[str, ...], str]:

    degree = len(wordlist[0])

    buckets = bucket(wordlist,
                     lambda _: (_[idx] for idx in range(0, degree, 2)))

    # buckets has value of an iterable, make it a list
    return {key: list(buckets[key]) for key in buckets}

def waffle_sample(wordlist: List[str]) -> List[str]:
    """
    Use rejection sampling and conditional probability
    to uniformly sample all valid word placements
    """

    # Now rejection sampling
    # Unfortunately random.choices uses replacement
    # we reject duplicates

    # Find horizontal
    degree = len(wordlist[0])
    rows = (degree + 1) // 2
    state = _word_buckets(wordlist)
    # Now choose columns
    tries = 0
    while True:
        tries += 1
        while True:
            horizontal = choices(wordlist, k=rows)
            if len(set(horizontal)) == rows:
                break
        keys = [tuple((word[idx] for idx in range(0, degree, 2)))
                for word in horizontal]
        if all((_ in state for _ in keys)):
            vertical = [state[_] for _ in keys]
            # We have a winner!
            print(f"Succeeded in {tries} tries.")
            return horizontal + [choice(_) for _ in vertical]

def _scoreit(state: Dict[Tuple[str, ...], str],
             horizontal: List[int]) -> float:
    
    degree = 2 * len(horizontal) - 1
    keys = [tuple((word[_] for _ in range(0, degree, 2)))
        for word in horizontal]
    addend = len(horizontal) - len(set(horizontal))
    return sum((_ not in state for _ in keys)) + addend

def metropolis(wordlist: List[str],
               temperature: float,
               trace: int = 0,
               burnin: int = 1000) -> List[str]:
    """
    Use the Metropolis algorithm to sample.
    """
    degree = len(wordlist[0])
    print(f"degree = {degree}")
    rows = (degree + 1) // 2
    state = _word_buckets(wordlist)
    # Initial population
    horizontal = choices(wordlist, k=rows)
    addend = len(horizontal) - len(set(horizontal))
    keys = [tuple((word[_] for _ in range(0, degree, 2)))
        for word in horizontal]
    score = sum((_ not in state for _ in keys)) + addend

    tries = 0
    taken = 0
    while True:
        tries += 1
        if trace > 0 and tries % trace == 0:
            print(f"Step {tries}, taken = {taken}, score = {score}")
        if tries > burnin and score == 0:
            vertical = [choice(state[_]) for _ in keys]
            print(f"Final tries = {tries}")
            return vertical + horizontal
        # modify horizontal
        new_horizontal = horizontal.copy()
        new_horizontal[randint(0, rows - 1)] = choice(wordlist)
        new_keys = [tuple((word[_] for _ in range(0, degree, 2)))
            for word in new_horizontal]
        new_addend = len(new_horizontal) - len(set(new_horizontal))
        new_score = sum((_ not in state for _ in new_keys)) + new_addend
        delta_score = new_score - score
        if (delta_score <= 0
            or exp(-delta_score/temperature) >= random()):
            # Accept this move
            score = new_score
            horizontal = new_horizontal
            taken += 1

class MetropolisBoard:
    def __init__(self, wordlist: List[str],
                 board: BOARD,
                 hamming: bool = True):

        self._wordlist = wordlist.copy()
        self._board = board.copy()
        self._subscore = self._hamming if hamming else self._basic
        if hamming:
            self._tree = KDTree(
                wordlist_to_vec(self._wordlist).toarray(),
                leaf_size=30, metric='manhattan')
        else:
            self._lookup = set(self._wordlist)
        self._assign = None

    def _basic(self, words: List[str]) -> float:
        return sum((_ not in self._lookup for _ in words))

    def _hamming(self, words: List[str]) -> float:

        distances, _ = self._tree.query(
            wordlist_to_vec(words).toarray(),
            k=1, return_distance=True)
        return 0.5 * distances.sum()

    def _score(self) -> float:
        words = [''.join((self._assign[_] for _ in elt))
            for elt in self._board]
        addend = len(words) - len(set(words))
        return addend + self._subscore(words)

    def sample(self,
               temperature: float,
               burnin: int = 1000,
               trace: int = 0):

        squares = list(sorted(set(chain(*self._board))))
        # initial guess
        self._assign = dict(zip(squares,
            choices(ascii_lowercase, k = len(squares))))
        tries = 0
        taken = 0
        score = self._score()

        while True:
            tries += 1
            if trace > 0 and tries % trace == 0:
                print(f"Step {tries}, taken {taken}, score = {score}")
            if tries > burnin and score == 0.0:
                return self._assign

            # Choose a random square and change it to a random
            # letter
            where = choice(squares)
            old = self._assign[where]
            how = old
            while True:
                how = choice(ascii_lowercase)
                if how != old:
                    break
            self._assign[where] = how
            new_score = self._score()
            delta_score = new_score - score
            if (delta_score <= 0
                or delta_score <= - temperature * log(random())):
                # Accept this move
                score = new_score
                taken += 1
            else:
                # reject, and put back the old state
                self._assign[where] = old 
