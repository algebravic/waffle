"""
Construct a random Waffle Game
"""
from typing import List, Tuple, Dict, Set, Iterable
from random import choice, choices, randint, random, Random
from math import exp, log
from string import ascii_lowercase
from itertools import chain
from more_itertools import bucket
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import KDTree
import pycmsgen
from .clue import BOARD, SQUARE
from .waffle_solve import Waffle

def wordlist_to_vec(wordlist: List[str]):
    """
    Convert a wordlist to a 1-hot sparse encoding.

    Each row corresponds to a word.
    There are 26 * m columns where m is the word length
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


def letter_status(wordlist: List[str], board: BOARD) -> Dict[SQUARE,
                                                             np.ndarray]:
    """
    Calculate monographic distributions for each square

    For each square, first find the pair (continuent, position)
    that the square can be in.  For each one of these
    use the monographic statistics, taking the geometric mean
    over all such pairs.
    """
    degree = len(wordlist[0])
    num = len(wordlist)
    squares = set(chain(*board))
    counters = [Counter((word[idx] for word in wordlist))
                for idx in range(degree)]
    # Laplace's law of succession
    rprobs = [1 + np.array([ctr[ltr] for ltr in ascii_lowercase])
        for ctr in counters]
    probs = [_ / _.sum() for _ in rprobs]
    where = ((pidx, idx, square)
        for pidx, place in enumerate(board)
        for idx, square in enumerate(place))
    buckets = bucket(where, lambda _: _[2])
    out = {}
    for square in squares:
        lprobs = np.array([probs[idx]
            for _, idx, _ in buckets[square]])
        out[square] = lprobs.prod(axis=0) ** (1.0 / lprobs.shape[0])
    return out

def entropy(probs: np.ndarray) -> float:
    return - (np.log(probs) * probs).sum()

class MetropolisBoard:
    def __init__(self, wordlist: List[str],
                 board: BOARD,
                 distribute: bool = False,
                 hamming: bool = True,
                 use_entropy: bool = False,
                 seed: int | None = None):

        self._random = random.Random()
        self._random.seed(seed)
        self._wordlist = wordlist.copy()
        self._board = board.copy()
        self._subscore = self._hamming if hamming else self._basic
        if hamming:
            self._tree = KDTree(
                wordlist_to_vec(self._wordlist).toarray(),
                leaf_size=30, metric='manhattan')
        else:
            self._lookup = set(self._wordlist)
        self._distribute = distribute
        self._squares = list(sorted(set(chain(*board))))
        if distribute:
            self._distr = letter_status(wordlist, board)
        else: # Uniform distribution
            unif = np.ones(len(ascii_lowercase)) / len(ascii_lowercase)
            self._distr = {square: unif for square in self._squares}
        self._assign = None
        # Choose square of highest entropy
        if use_entropy:
            wgts = [entropy(self._distr[square])
                for square in self._squares]
            self._weights = [_/ sum(wgts) for _ in wgts]
        else:
            self._weights = [1/len(self._squares)
                for _ in self._squares]

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

        # initial guess
        self._assign = {square: self._random.choices(
            ascii_lowercase,
            weights = self._distr[square])[0]
            for square in self._squares}
        tries = 0
        taken = 0
        score = self._score()

        back = {val: key for key, val in enumerate(ascii_lowercase)}

        while True:
            tries += 1
            if trace > 0 and tries % trace == 0:
                print(f"Step {tries}, taken {taken}, score = {score}")
            if tries > burnin and score == 0.0:
                return self._assign

            # Choose a random square and change it to a random
            # letter
            where = self._random.choices(self._squares, weights = self._weights)[0]
            old = self._assign[where]
            how = old
            while True:
                how = self._random.choices(ascii_lowercase,
                    weights = self._distr[where])[0]
                if how != old:
                    break
            self._assign[where] = how
            new_score = self._score()
            if self._distribute: # Non reversible chain
                # fix up the score
                # N = # of squares
                # pi(y) = exp(-E(y)/T) / Z
                # K(x,y) = 1/N * Prob(new letter)
                # Prob(new letter) = P(new)/(1-Prob(old))
                tab = self._distr[where]
                t_how = tab[back[how]]
                t_old = tab[back[old]]
                numer = t_old * (1.0 - t_old)
                denom = t_how * (1.0 - t_how)
                supp = log(numer / denom) * temperature
            else:
                supp = 0.0 # Reversible
            delta_score = new_score - score + supp
            if (delta_score <= 0
                or delta_score <= - temperature * log(random())):
                # Accept this move
                score = new_score
                taken += 1
            else:
                # reject, and put back the old state
                self._assign[where] = old

    def word_sample(self,
                    temperature: float,
                    burnin: int = 1000,
                    trace: int = 0):
        values = self.sample(temperature,
            burnin = burnin,
            trace = trace)

        # Convert to words
        return [''.join((values[_] for _ in elt))
                for elt in self._board]

def cms_sample(waf: Waffle, seed: int | None = None) -> Iterable[Tuple[str, 0]]:

    board = waf._board
    solver = pycmsgen.Solver(seed = seed)
    waf._basic()
    solver.add_clauses(waf._cnf.clauses)
    while True:
        status = solver.solve()
        if not status:
            break
        pos = [waf._pool.obj(_) for _ in solver.get_model() if _ > 0]
        yield [_[1: ] for _ in pos if _ is not None and _[0] == 'w']
