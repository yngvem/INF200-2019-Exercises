# -*- coding: utf-8 -*-

"""
Exercise 01-B. Convert a list comprehension to a loop.
"""

__author__ = 'Hans Ekkehard Plesser'
__email__ = 'hans.ekkehard.plesser@nmbu.no'


def squares_by_loop(n):
    res = []
    for k in range(n):
        if k % 3 == 1:
            res.append(k ** 2)
    return res


def squares_by_comp(n):
    return [k ** 2 for k in range(n) if k % 3 == 1]


if __name__ == '__main__':
    limit = 20
    print(squares_by_loop(limit))
    print(squares_by_comp(limit))
    if squares_by_loop(limit) != squares_by_comp(limit):
        print("ERROR!")
