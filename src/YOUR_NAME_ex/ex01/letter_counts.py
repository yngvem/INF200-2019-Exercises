# -*- coding: utf-8 -*-

"""
Exercise 01-C. Count the frequency of the various letters in a string.
"""

import collections

__author__ = 'Hans Ekkehard Plesser'
__email__ = 'hans.ekkehard.plesser@nmbu.no'


def letter_freq(string):
    """
    Count the number of times different character appears in a string.
    """

    counts = {}
    for char in string.lower():
        if char in counts:
            counts[char] += 1
        else:
            counts[char] = 1
    return counts 


def letter_freq_get(string):
    """
    Count letter frequency, using dict.get() to default missing keys.
    """

    counts = {}
    for char in string.lower():
        counts[char] = counts.get(char, 0) + 1
    return counts


def letter_freq_defaultdict(string):
    """
    Count letter frequency, use collection.defaultdict to fill initial values.
    """

    counts = collections.defaultdict(int)
    for char in string.lower():
        counts[char] += 1
    return counts


def letter_freq_str_count(string):
    """
    Count letter frequency, use str.count() method.
    """

    lc_string = string.lower()
    return {char: lc_string.count(char) for char in set(lc_string)}


def letter_freq_counter(string):
    """
    Count letter frequency, use collections.Counter().
    """

    # noinspection PyArgumentList
    return collections.Counter(string.lower())


if __name__ == '__main__':
    text = input('Please enter text to analyse: ')

    # ensure all approaches obtain the same result
    assert letter_freq(text) == letter_freq_get(text)
    assert letter_freq(text) == letter_freq_defaultdict(text)
    assert letter_freq(text) == letter_freq_str_count(text)
    assert letter_freq(text) == letter_freq_counter(text)

    frequencies = letter_freq(text)
    for letter, count in frequencies.items():
        print('{:3}{:10}'.format(letter, count))

    print("\n\nAnd sorted ...")
    for letter, count in sorted(frequencies.items()):
        print('{:3}{:10}'.format(letter, count))
