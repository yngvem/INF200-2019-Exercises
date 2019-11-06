# -*- coding: utf-8 -*-

"""
Minimal set of compatibility tests for PA02.
"""

__author__ = 'Hans Ekkehard Plesser'
__email__ = 'hans.ekkehard.plesser@nmbu.no'


import chutes_simulation as cs
import pytest

class TestBoard:
    """
    Tests for Board class.
    """

    def test_constructor_default(self):
        """Default constructor callable."""
        b = cs.Board()
        assert isinstance(b, cs.Board)

    def test_constructor_args(self):
        """Constructor with unnamed arguments callable."""
        b = cs.Board([(1, 4), (5, 16)], [(9, 2), (12, 3)], 90)
        assert isinstance(b, cs.Board)

    def test_constructor_named_args(self):
        """Constructor with kw args callable."""
        b = cs.Board(ladders=[(1, 4), (5, 16)], chutes=[(9, 2), (12, 3)], goal=90)
        assert isinstance(b, cs.Board)

    def test_goal_reached(self):
        """goal_reached() callable and returns bool"""
        b = cs.Board()
        assert isinstance(b.goal_reached(1), bool)

    def test_position_adjustment(self):
        """position_adjustment callable and returns number"""
        b = cs.Board()
        assert isinstance(b.position_adjustment(1), (int, float))


class TestPlayer:
    """
    Tests for Player class.
    """

    def test_constructor(self):
        """Player can be constructed."""
        b = cs.Board()
        p = cs.Player(b)
        assert isinstance(p, cs.Player)

    def test_move(self):
        """Player has move() method."""
        b = cs.Board()
        p = cs.Player(b)
        p.move()


class TestResilientPlayer:
    def test_constructor(self):
        """ResilientPlayer can be created."""
        b = cs.Board()
        p = cs.ResilientPlayer(b, extra_steps=4)
        assert isinstance(p, cs.ResilientPlayer)
        assert isinstance(p, cs.Player)

    def test_move(self):
        """ResilientPlayer can move."""
        b = cs.Board()
        p = cs.ResilientPlayer(b)
        p.move()


class TestLazyPlayer:
    def test_constructor(self):
        """LazyPlayer can be constructed."""
        b = cs.Board()
        p = cs.LazyPlayer(b, dropped_steps=3)
        assert isinstance(p, cs.LazyPlayer)
        assert isinstance(p, cs.Player)

    def test_move(self):
        """LazyPlayer can move."""
        b = cs.Board()
        p = cs.LazyPlayer(b)
        p.move()


class TestSimulation:
    """Tests for Simulation class."""

    def test_constructor_default(self):
        """Default constructor works."""
        s = cs.Simulation([cs.Player, cs.Player])
        assert isinstance(s, cs.Simulation)

    def test_constructor_named(self):
        """Constructor with kw args works."""
        b = cs.Board()
        s = cs.Simulation(player_field=[cs.Player, cs.Player],
                          board=b, seed=123, randomize_players=True)
        assert isinstance(s, cs.Simulation)

    def test_single_game(self):
        """single_game() returns non-negative number and class name"""
        s = cs.Simulation([cs.Player, cs.Player])
        nos, wc = s.single_game()
        assert nos > 0
        assert wc == 'Player'

    def test_run_simulation(self):
        """run_simulation() can be called"""
        s = cs.Simulation([cs.Player, cs.Player])
        s.run_simulation(2)

    def test_simulation_results(self):
        """
        - Multiple calls to run_simulation() aggregate results
        - get_results() returns list of result tuples
        """
        s = cs.Simulation([cs.Player, cs.Player])
        s.run_simulation(2)
        r = s.get_results()
        assert len(r) == 2
        s.run_simulation(1)
        r = s.get_results()
        assert len(r) == 3
        assert all(s > 0 and t == 'Player' for s, t in r)

    def test_players_per_type(self):
        """player_per_type() returns dict mapping names to non-neg numbers."""
        s = cs.Simulation([cs.Player, cs.LazyPlayer, cs.ResilientPlayer])
        p = s.players_per_type()
        assert all(k in ['Player', 'LazyPlayer', 'ResilientPlayer']
                   for k in p.keys())
        assert all(v >= 0 for v in p.values())

    def test_winners_per_type(self):
        """winners_per_type() returns dict mapping names to non-neg numbers."""
        s = cs.Simulation([cs.Player, cs.LazyPlayer, cs.ResilientPlayer])
        s.run_simulation(10)
        w = s.winners_per_type()
        assert all(k in ['Player', 'LazyPlayer', 'ResilientPlayer']
                   for k in w.keys())
        assert all(v >= 0 for v in w.values())

    def test_durations_per_type(self):
        """
        durations_per_type() returns dict mapping names to list of
        non-neg numbers.
        """
        s = cs.Simulation([cs.Player, cs.LazyPlayer, cs.ResilientPlayer])
        s.run_simulation(10)
        w = s.durations_per_type()
        assert all(k in ['Player', 'LazyPlayer', 'ResilientPlayer']
                   for k in w.keys())
        assert all(len(v) >= 0 for v in w.values())
        assert all(n >= 0 for v in w.values() for n in v)
