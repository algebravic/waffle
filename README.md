The Waffle game
===============

The Waffle game [[https://wafflegame.net/daily][Waffle]] is an
internet game.  Each challenge consists of 5 uknown 5 letter words
arranged in a 5 by 5 square, covering the top, bottom, left, right and
middle (there are 4 squares that are unoccupied).  Each occupied
square is filled with a letter, with a a color.  The answer is
obtained by some unknown permutation (which is a product of 5
transpositions) of the filled squares.  The color code gives clues:
green indicates that the letter is in a correct position.  Yellow
indicates that the row/column containing that square contains that
letter, but not in that position (more detail later).  Black indicates
that the row/column does not contain that letter, and not in that
position.  There's an extra condition if more than on square in a
row/column contains the same letter.  The rules are not completely
clear on this but I believe the following is true:

An *internal* square is one that is contained in exactly one
row/column.  If an internal square contains a yellow marked letter,
that indicates that that letter must appear somewhere else in that
row/column. By contrast, an external square only implies that it must
appear in the contained row or column.

