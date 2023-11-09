"""
Form a Huffman code from monographic statistics, do encoding
and decoding.
"""
from typing import List, Tuple, Dict, Set, Iterable
from collections import Counter
from itertools import chain
from heapq import heappop, heappush, heapify
import nltk.corpus
from .get_words import DATADIR, get_words

CODE = Tuple[int, int]

NODE = Tuple[float, Tuple['NODE', 'NODE'] | str]

class Node:

    def __init__(self, prob: float, val: str | Tuple['Node', 'Node']):

        self._prob = prob
        self._val = val

    def __le__(self, other):
        return self._prob <= other._prob

    def __lt__(self, other):
        return self._prob < other._prob

def produce_bits(root: Node) -> Iterable[Tuple[str, str]]:
    """
    Descend the Huffman tree, producing encodings.
    """
    if isinstance(root._val, str):
        yield (root._val, '')
    else:
        lft, rgt = root._val
        yield from ((_[0], '0' + _[1]) for _ in produce_bits(lft))
        yield from ((_[0], '1' + _[1]) for _ in produce_bits(rgt))

def produce_table(stats: Dict[str, float]) -> Dict[str, CODE]:
    """
    Input: A dictionary keyed by string, whose values
    are nonnegative floats giving the relative frequency

    Output:
    A huffman code keyed by the same keys as the input
    """
    table = [Node(prob, key) for key, prob in stats.items()]
    heapify(table)
    while len(table) > 1:
        zero = heappop(table)
        one = heappop(table)
        heappush(table, Node(zero._prob + one._prob, (zero, one)))
    # Now table[0] is the root of the tree
    return table[0]

def get_stats(words: Iterable[str]) -> Dict[str, float]:
    """
    Get word probabilities
    Input: words
    Output: A dictionary keyed by the words with values
       relative frequency.
    We add the 'terminal' word xxxx, to indicate end of message.
    """
    count = Counter(chain(words, ['xxxx']))
    denom = sum(count.values())
    return {key: val / denom for key, val in count.items()}

def get_letter_stats(words: Iterable[str]) -> Dict[str, float]:

    return get_stats(chain(*words))

def encode(table: Dict[str, str], word: str | List[str]) -> str:
    """
    Input:
      table: a Huffman code table.
      word: either a string or List of strings
    Ouput:
      The Huffman code for the input as a 0/1 string.
    """

    return ''.join((table[_] for _ in word))

def decode_one(root: Node, code: str) -> Tuple[str | None, str]:
    """
    Decode on element in a Huffman coded string
    Input:
      root: a Huffman code tree
    Output:
      An iterable of the decoded constituents
      followed by the reamining undecoded string
      because it was an incomplete code word.
      If empty it indicates complete decoding.
    """
    if isinstance(root._val, str):
        return root._val, code
    elif len(code) == 0:
        return None, ''
    else:
        idx = '01'.find(code[0])
        if idx == -1:
            raise ValueError(f"Illegal code {code[0]}")
        res, rest = decode_one(root._val[idx], code[1:])
        if res is None:
            return None, code
        else:
            return res, rest
    
def decode(root: Node, code: str) -> Iterable[str]:
    """
    Decode a Huffman coded string.
    """
    tcode = code
    while True:
        res, rest = decode_one(root, tcode)
        if res is None:
            yield rest
            return
        else:
            yield res
            tcode = rest
            
def corpus_code(corpus_name: str = 'brown', categories='news') -> Tuple[Dict[str, CODE], Node]:
    """
    Get a corpus and category and produce the huffman table.
    """
    corpus = getattr(nltk.corpus, corpus_name)
    words = corpus.words(categories=categories)
    stats = get_stats((_.lower() for _ in words))
    tree = produce_table(stats)
    return dict(produce_bits(tree)), tree
