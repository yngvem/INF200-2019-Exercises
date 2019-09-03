# -*- coding: utf-8 -*-

"""
Exercise 01-D. Tidied version of script.
"""

from random import randint

__author__ = 'Hans Ekkehard Plesser'
__email__ = 'hans.ekkehard.plesser'


def user_guess():
    guess = 0
    while guess < 1:
        guess = int(input('Your guess: '))
    return guess


def roll_two_dice():
    return randint(1, 6) + randint(1, 6)


def equal(value, guess):
    return value == guess


if __name__ == '__main__':

    correct_answer = False
    points_remaining = 3
    number_to_guess = roll_two_dice()
    while not correct_answer and points_remaining > 0:
        number_guessed = user_guess()
        correct_answer = equal(number_to_guess, number_guessed)
        if not correct_answer:
            print('Wrong, try again!')
            points_remaining -= 1

    if points_remaining > 0:
        print('You won {} points.'.format(points_remaining))
    else:
        print('You lost. Correct answer: {}.'.format(number_to_guess))
