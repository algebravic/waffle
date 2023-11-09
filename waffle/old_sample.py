from typing import List, Tuple, Dict, Set, Iterable
from random import choice, choices, randint, random, Random
from math import exp, log
from string import ascii_lowercase
from itertools import chain
from more_itertools import bucket

def _scoreit(state: Dict[Tuple[str, ...], str],
             horizontal: List[int]) -> float:
    
    degree = 2 * len(horizontal) - 1
    keys = [tuple((word[_] for _ in range(0, degree, 2)))
        for word in horizontal]
    addend = len(horizontal) - len(set(horizontal))
    return sum((_ not in state for _ in keys)) + addend

def _word_buckets(wordlist: List[str]) -> Dict[Tuple[str, ...], str]:

    degree = len(wordlist[0])

    buckets = bucket(wordlist,
                     lambda _: (_[idx] for idx in range(0, degree, 2)))

    # buckets has value of an iterable, make it a list
    return {key: list(buckets[key]) for key in buckets}

def metropolis(wordlist: List[str],
               temperature: float,
               distribute: bool = False,
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

