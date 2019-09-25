# -*- coding: utf-8 -*-

__author__ = "Yngve Mardal Moe"
__email__ = "yngve.m.moe@gmail.com"


def squares_by_comp(n):
    return [k**2 for k in range(n) if k % 3 == 1]


def squares_by_loop(n):
    data = []
    for k in range(n):
        if k % 3 == 1:
            data.append(k**2)
    return data


if __name__ == '__main__':
    if squares_by_loop(20) != squares_by_comp(20):
        print('ERROR!')
