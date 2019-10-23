# -*- coding: utf-8 -*-

"""
Acceptance test suite for EX05.

Your code should pass these tests before submission.
"""

from walker_sim import Walker, Simulation
from bounded_sim import BoundedWalker, BoundedSimulation
from myrand import LCGRand

__author__ = "Hans Ekkehard Plesser"
__email__ = "hans.ekkehard.plesser@nmbu.no"


def test_lcg():
    """Test that LCG generator works."""

    lcg = LCGRand(346)
    assert lcg.rand() == 5815222
    assert lcg.rand() == 1099672039


def test_rand_iter():
    """Test that the iterator works."""
    lcg = LCGRand(3)
    true = [50421, 847425747, 572982925]
    for rand, target in zip(lcg.random_sequence(3), true):
        assert rand == target


def test_infinite_numbers():
    """Test that the infinite generator works."""
    lcg = LCGRand(3)
    true = [50421, 847425747, 572982925, 807347327, 1284843143, 1410633816]
    for i, (rand, target) in enumerate(
        zip(lcg.infinite_random_sequence(), true)
    ):
        assert rand == target, "The random number was incorrect"
    assert i == len(true) - 1, "The sequence stopped by itself"

    for i, target in enumerate(lcg.infinite_random_sequence()):
        if i > 100:
            break
    else:
        assert False, "The random sequence stopped by itself"


def test_walker():
    """Test that Walker class can be used as required."""

    start, home = 10, 20
    w = Walker(start, home)
    assert not w.is_at_home()
    w.move()
    assert w.get_position() != start
    w.move()
    assert w.get_steps() == 2


def test_simulation():
    """Test that Simulation class can be used as required."""

    start, home, seed, n_sim = 10, 20, 12345, 5
    s = Simulation(start, home, seed)
    assert s.single_walk() > 0
    r = s.run_simulation(n_sim)
    assert len(r) == n_sim
    assert all(rs > 0 for rs in r)


def test_bounded_walker():
    """Test that BoundedWalker class can be used as required."""

    start, home, left, right = 10, 20, 0, 30
    w = BoundedWalker(start, home, left, right)
    assert not w.is_at_home()
    w.move()
    assert w.get_position() != start
    w.move()
    assert w.get_steps() == 2


def test_bounded_simulation():
    """Test that BoundedSimulation class can be used as required."""

    start, home, left, right, seed, n_sim = 10, 20, 0, 30, 12345, 5
    s = BoundedSimulation(start, home, seed, left, right)
    assert s.single_walk() > 0
    r = s.run_simulation(n_sim)
    assert len(r) == n_sim
    assert all(rs > 0 for rs in r)


def test_gambler():
    """Test that Gambler class can be used as required."""

    initial, total, p = 50, 100, 0.49
    g = Gambler(initial, total, p)
    assert not g.is_broke()
    assert not g.owns_all()
    g.play()
    assert not g.is_broke()
    assert not g.owns_all()


def test_gambler_simulation():
    """Test that GamblerSimulation class can be used as required."""

    initial, total, p, seed, n_sim = 50, 100, 0.49, 12345, 5
    c = GamblerSimulation(initial, total, p, seed)
    res, n = c.single_game()
    assert res in [True, False]
    assert n > 0

    n_win, n_loss = c.run_simulation(n_sim)
    assert len(n_win) + len(n_loss) == n_sim
    assert all(n > 0 for n in n_win)
    assert all(n > 0 for n in n_loss)
