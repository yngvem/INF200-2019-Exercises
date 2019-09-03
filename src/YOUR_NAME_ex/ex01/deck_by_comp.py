# -*- coding: utf-8 -*-

"""
Exercise 01-A. Create a deck of cards using list comprehension.
"""

__author__ = 'Hans Ekkehard Plesser'
__email__ = 'hans.ekkehard.plesser@nmbu.no'

SUITS = ('C', 'S', 'H', 'D')
VALUES = range(1, 14)


def deck_loop():
    deck = []
    for suit in SUITS:
        for val in VALUES:
            deck.append((suit, val))
    return deck


def deck_comp():
    return [(s, v) for s in SUITS for v in VALUES]


if __name__ == '__main__':
    if deck_loop() != deck_comp():
        print("ERROR!")
