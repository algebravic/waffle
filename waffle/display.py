"""
Use matplotlib to display a Waffle Board.
"""

from typing import Tuple, List, Set, Iterable, Dict
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from itertools import chain
import numpy as np
from .reconstruct import square_to_board, assign_colors
from .clue import COLOR, CLUES, BOARD, get_clues
from .waffle_solve import PLACEMENT

def render_clues(clues: CLUES,
                 size: float = 120.0,
                 border: float = 0.2,
                 corner_radius: float = 0.05):

    board = square_to_board(list(clues.keys()))
    
    min_x = min((_[0] for _ in clues.keys()))
    max_x = max((_[0] for _ in clues.keys()))
    x_span = max_x - min_x + 1
    min_y = min((_[1] for _ in clues.keys()))
    max_y = max((_[1] for _ in clues.keys()))
    y_span = max_y - min_y + 1
        

    x_width = 0.5 * (size / x_span)
    x_bound = np.linspace(0.0, size, num = x_span + 1)
    x_centers = 0.5 * (x_bound[1:] + x_bound[:-1])
    y_bound = np.linspace(0.0, 2 * x_width * y_span, num = y_span + 1)
    y_centers = 0.5 * (y_bound[1:] + y_bound[:-1])
    pad = x_width * border
    dim = 2 * x_width  - 2 * pad
    fig, ax = plt.subplots()

    radius = sqrt(2.0) * x_width * corner_radius

    squares = {}
    colors = {COLOR.yellow: 'orange',
        COLOR.green: 'green',
        COLOR.black:'white'}
    texts = {COLOR.yellow: 'white',
        COLOR.green: 'white',
        COLOR.black:'black'}
    for square, (letter, color) in clues.items():
        # (x,y) are lower left corner
        center = (x_centers[square[0]], y_centers[square[1]])
        ax.text(center[1] - pad, - center[0] - pad,
                letter.upper(),
                ha='center', va='center',
                fontsize=20,
                color=texts[color])

        llc = (center[1] - x_width, - center[0] - x_width)
        squares[square] = FancyBboxPatch(
            llc,
            dim, dim,
            boxstyle = f"round,pad={radius}",
            ec='black',
            fc= colors[color],
            lw=2)
    
    list(map(ax.add_patch, squares.values()))

    extra = x_width * border
    ax.set_xlim(-extra, size + extra)
    ax.set_ylim(- 2 * x_width * y_span - extra , extra)
    plt.grid(False)
    plt.axis('off')
    return fig, ax

def write_clues(name: str, board: BOARD):
    fig, ax = render_clues(get_clues(name, board))
    plt.savefig(f'waffle_{name}.png')

def display_state(name: str,
                  current: PLACEMENT,
                  solution: PLACEMENT):

    the_clues = assign_colors(current, solution)
    fig, ax = render_clues(the_clues)
    plt.savefig(f'waffle_{name}.png')
