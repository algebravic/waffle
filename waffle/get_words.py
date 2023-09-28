"""
Get words.
"""
from typing import Iterable

def system_words(size: int) -> Iterable[str]:
    with open('/usr/share/dict/words', 'r') as myfile:
        trunc = (_[:-1] for _ in myfile.readlines())
        yield from (_ for _ in trunc if len(_) == size
                    and _[0].islower())
