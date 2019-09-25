# -*- coding: utf-8 -*-

__author__ = "Yngve Mardal Moe"
__email__ = "yngve.m.moe@gmail.com"

from random import randint


def guess_dice_roll():
    guess = 0
    while guess < 1:
        guess = int(input('Your guess: '))
    return guess


def roll_two_dice():
    return randint(1, 6) + randint(1, 6)


def check_answer(guess, answer):
    return guess == answer


if __name__ == '__main__':
    has_won = False
    attempts_left = 3
    answer = roll_two_dice()

    while not has_won and attempts_left > 0:
        guess = guess_dice_roll()

        has_won = check_answer(guess, answer)
        if not has_won:
            print("Wrong, try again!")
            attempts_left -= 1

    if attempts_left > 0:
        print(f"You won {attempts_left} points.")
    else:
        print(f"You lost. Correct answer: {answer}.")
