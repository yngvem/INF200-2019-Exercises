# -*- coding: utf-8 -*-

__author__ = "Yngve Mardal Moe"
__email__ = "yngve.m.moe@gmail.com"

from collections import Counter, defaultdict


def letter_freq(txt):
    # Best solution
    return Counter(txt)

    # Next best solution
    frequencies = defaultdict(int)
    for letter in txt:
        frequencies[letter] += 1
    return frequencies

    # Alternatively
    frequencies = {}
    for letter in txt:
        if letter not in frequencies:
            frequencies[letter] = 1
        else:
            frequencies[letter] += 1

    return frequencies


if __name__ == '__main__':
    text = input('Please enter text to analyse: ')

    frequencies = letter_freq(text)
    for letter, count in frequencies.items():
        print('{:3}{:10}'.format(letter, count))