"""
Get words.
"""
from typing import Iterable, List
from itertools import chain

DATADIR = 'data/dictionaries'

def get_words(data: str = 'wordlist', wlen: int = 5) -> Iterable[str]:
    """
    Get the word list.
    """
    fname = ('/usr/share/dict/words' if data == 'system'
             else f"{DATADIR}/{data}.txt")
    with open(fname) as myfile:
        yield from ((_[: -1].lower() for _ in myfile.readlines()
                    if len(_[: -1]) == wlen and _[: - 1].isalpha()))

def system_words(size: int) -> Iterable[str]:
    with open('/usr/share/dict/words', 'r') as myfile:
        trunc = (_[:-1] for _ in myfile.readlines())
        yield from (_ for _ in trunc if len(_) == size
                    and _[0].islower())

def merge_words(size: int, files: List[str], out: str):
    """
    Merge and sort words of a given size.
    """
    aggregate = sorted(set(chain(*(get_words(_, size) for _ in files))))
    with open(f"data/{out}.txt", 'w') as myfile:
        myfile.write('\n'.join(aggregate))
        myfile.write('\n')
