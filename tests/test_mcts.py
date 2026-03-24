"""Tests for MCTS search and player integration."""

import random

import numpy as np
import pytest

from blokus.board import NUM_PLAYERS, STARTING_CORNERS
from blokus.game import Game
from blokus.pieces import PIECES
from blokus.players import MCTSPlayer, RandomPlayer, play_game
from mcts.search import MCTS, MCTSNode, _normalize_scores


# ------------------------------------------------------------------ #
# Score normalisation
# ------------------------------------------------------------------ #

class TestNormalization:
    def test_min_score(self):
        normed = _normalize_scores([-89, -89, -89, -89])
        np.testing.assert_allclose(normed, [0, 0, 0, 0])

    def test_max_score(self):
        normed = _normalize_scores([20, 20, 20, 20])
        np.testing.assert_allclose(normed, [1, 1, 1, 1])

    def test_zero(self):
        normed = _normalize_scores([0, 0, 0, 0])
        expected = 89 / 109
        np.testing.assert_allclose(normed, [expected] * 4)


# ------------------------------------------------------------------ #
# MCTSNode basics
# ------------------------------------------------------------------ #

class TestMCTSNode:
    def test_initial_state(self):
        node = MCTSNode(parent=None, move=None, player=0)
        assert node.visits == 0
        assert not node.is_terminal
        assert not node.is_fully_expanded  # untried_moves is None

    def test_fully_expanded(self):
        node = MCTSNode(parent=None, move=None, player=0)
        node.untried_moves = []
        assert node.is_fully_expanded

    def test_not_fully_expanded(self):
        node = MCTSNode(parent=None, move=None, player=0)
        node.untried_moves = ["some_move"]
        assert not node.is_fully_expanded

    def test_mean_value(self):
        node = MCTSNode(parent=None, move=None, player=0)
        node.visits = 10
        node.value_sum = np.array([5.0, 3.0, 2.0, 0.0])
        assert node.mean_value(0) == pytest.approx(0.5)
        assert node.mean_value(1) == pytest.approx(0.3)


# ------------------------------------------------------------------ #
# MCTS search correctness
# ------------------------------------------------------------------ #

class TestMCTSSearch:
    def test_returns_legal_move(self):
        """MCTS must return a move that's in the legal move set."""
        game = Game()
        mcts = MCTS(num_simulations=20, rollout_rng=random.Random(42))
        move, stats = mcts.search(game)
        legal = game.get_legal_moves()
        assert move in legal

    def test_first_move_covers_corner(self):
        """First move from MCTS must cover the starting corner."""
        game = Game()
        mcts = MCTS(num_simulations=50, rollout_rng=random.Random(7))
        move, _ = mcts.search(game)
        piece = PIECES[move.piece_name]
        ori = piece.orientations[move.orientation_idx]
        cells = {(move.row + dr, move.col + dc) for dr, dc in ori.cells}
        assert STARTING_CORNERS[0] in cells

    def test_stats_populated(self):
        game = Game()
        mcts = MCTS(num_simulations=30, rollout_rng=random.Random(1))
        _, stats = mcts.search(game)
        assert stats["simulations"] == 30
        assert stats["root_visits"] == 30
        assert stats["children"] > 0
        assert stats["elapsed_s"] > 0

    def test_deterministic_with_seed(self):
        """Same seed ⇒ same move."""
        game = Game()
        m1, _ = MCTS(num_simulations=50, rollout_rng=random.Random(99)).search(game)
        m2, _ = MCTS(num_simulations=50, rollout_rng=random.Random(99)).search(game)
        assert m1 == m2

    def test_mid_game_search(self):
        """MCTS works after several moves have been played."""
        rng = random.Random(55)
        game = Game()
        # Play 8 random moves
        for _ in range(8):
            if game.game_over:
                break
            game.apply_move(rng.choice(game.get_legal_moves()))

        if not game.game_over:
            mcts = MCTS(num_simulations=30, rollout_rng=random.Random(55))
            move, _ = mcts.search(game)
            assert move in game.get_legal_moves()


# ------------------------------------------------------------------ #
# Game copy
# ------------------------------------------------------------------ #

class TestGameCopy:
    def test_copy_independence(self):
        game = Game()
        moves = game.get_legal_moves()
        game.apply_move(moves[0])

        copy = game.copy()
        copy_moves = copy.get_legal_moves()
        copy.apply_move(copy_moves[0])

        # Original should be unaffected
        assert game.board.grid.sum() != copy.board.grid.sum()

    def test_copy_preserves_state(self):
        game = Game()
        game.apply_move(game.get_legal_moves()[0])
        copy = game.copy()
        assert copy.current_player == game.current_player
        assert copy.game_over == game.game_over
        assert len(copy.move_history) == len(game.move_history)


# ------------------------------------------------------------------ #
# Full-game integration
# ------------------------------------------------------------------ #

class TestPlayGame:
    def test_random_vs_random(self):
        players = [RandomPlayer(seed=i) for i in range(4)]
        game = play_game(players)
        assert game.game_over

    def test_mcts_vs_random_completes(self):
        """One MCTS player + 3 random — game must finish without errors."""
        players = [
            MCTSPlayer(num_simulations=10, seed=0),
            RandomPlayer(seed=1),
            RandomPlayer(seed=2),
            RandomPlayer(seed=3),
        ]
        game = play_game(players)
        assert game.game_over
        scores = game.get_scores()
        assert all(-89 <= s <= 20 for s in scores)
