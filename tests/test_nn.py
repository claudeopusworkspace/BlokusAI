"""Tests for neural network, encoding, and neural MCTS."""

import random

import numpy as np
import pytest
import torch

from blokus.board import BOARD_SIZE, NUM_PLAYERS, Move, STARTING_CORNERS
from blokus.game import Game
from blokus.pieces import PIECES, PIECE_NAMES
from nn.encoding import (
    ACTION_SPACE_SIZE,
    NUM_PAIRS,
    decode_action,
    encode_action,
    encode_legal_mask,
    encode_policy,
    normalize_scores_for_nn,
    nn_value_to_mcts,
)
from nn.model import BlokusNet
from mcts.neural_mcts import NeuralMCTS, NeuralMCTSNode


# ------------------------------------------------------------------ #
# Encoding
# ------------------------------------------------------------------ #

class TestEncoding:
    def test_action_space_size(self):
        assert NUM_PAIRS == 91
        assert ACTION_SPACE_SIZE == 36_400

    def test_roundtrip(self):
        """encode → decode recovers the original move."""
        move = Move("L4", 2, 5, 10)
        assert decode_action(encode_action(move)) == move

    def test_all_pieces_encodable(self):
        for name in PIECE_NAMES:
            piece = PIECES[name]
            for oi in range(len(piece.orientations)):
                move = Move(name, oi, 0, 0)
                idx = encode_action(move)
                assert 0 <= idx < ACTION_SPACE_SIZE
                assert decode_action(idx) == move

    def test_legal_mask_shape(self):
        game = Game()
        mask = encode_legal_mask(game.get_legal_moves())
        assert mask.shape == (ACTION_SPACE_SIZE,)
        assert mask.dtype == np.bool_
        assert mask.sum() == len(game.get_legal_moves())

    def test_encode_policy(self):
        probs = {0: 0.3, 100: 0.7}
        dense = encode_policy(probs)
        assert dense.shape == (ACTION_SPACE_SIZE,)
        assert dense[0] == pytest.approx(0.3)
        assert dense[100] == pytest.approx(0.7)
        assert dense.sum() == pytest.approx(1.0)


class TestScoreNormalization:
    def test_range(self):
        normed = normalize_scores_for_nn([-89, 20, 0, -40])
        assert all(-1.0 <= v <= 1.0 for v in normed)

    def test_endpoints(self):
        normed = normalize_scores_for_nn([-89, 20, -89, 20])
        np.testing.assert_allclose(normed, [-1, 1, -1, 1], atol=1e-6)

    def test_mcts_conversion(self):
        nn_val = np.array([-1.0, 0.0, 1.0, 0.5])
        mcts_val = nn_value_to_mcts(nn_val)
        np.testing.assert_allclose(mcts_val, [0.0, 0.5, 1.0, 0.75])


# ------------------------------------------------------------------ #
# Network
# ------------------------------------------------------------------ #

class TestBlokusNet:
    @pytest.fixture
    def net(self):
        return BlokusNet(num_filters=16, num_blocks=2)

    def test_output_shapes(self, net):
        x = torch.randn(4, 10, 20, 20)
        policy, value = net(x)
        assert policy.shape == (4, ACTION_SPACE_SIZE)
        assert value.shape == (4, NUM_PLAYERS)

    def test_value_range(self, net):
        x = torch.randn(2, 10, 20, 20)
        _, value = net(x)
        assert (value >= -1).all() and (value <= 1).all()

    def test_single_example(self, net):
        game = Game()
        state = torch.from_numpy(game.get_state_planes()).unsqueeze(0)
        policy, value = net(state)
        assert policy.shape == (1, ACTION_SPACE_SIZE)
        assert value.shape == (1, NUM_PLAYERS)

    def test_param_count(self, net):
        total = sum(p.numel() for p in net.parameters())
        assert total > 0
        assert total < 5_000_000  # sanity: not absurdly large


# ------------------------------------------------------------------ #
# Neural MCTS
# ------------------------------------------------------------------ #

class TestNeuralMCTS:
    @pytest.fixture
    def net(self):
        return BlokusNet(num_filters=16, num_blocks=2)

    def test_returns_legal_move(self, net):
        game = Game()
        mcts = NeuralMCTS(net, num_simulations=10)
        move, probs = mcts.search(game, temperature=1.0, add_noise=False)
        assert move in game.get_legal_moves()

    def test_action_probs_sum_to_one(self, net):
        game = Game()
        mcts = NeuralMCTS(net, num_simulations=20)
        _, probs = mcts.search(game, temperature=1.0, add_noise=False)
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_first_move_covers_corner(self, net):
        game = Game()
        mcts = NeuralMCTS(net, num_simulations=10)
        move, _ = mcts.search(game, temperature=0.0, add_noise=False)
        piece = PIECES[move.piece_name]
        ori = piece.orientations[move.orientation_idx]
        cells = {(move.row + dr, move.col + dc) for dr, dc in ori.cells}
        assert STARTING_CORNERS[0] in cells

    def test_full_game_completes(self, net):
        """Neural MCTS can play a full game without crashing."""
        game = Game()
        mcts = NeuralMCTS(net, num_simulations=5)
        moves_played = 0
        while not game.game_over and moves_played < 200:
            move, _ = mcts.search(game, temperature=1.0, add_noise=False)
            game.apply_move(move)
            moves_played += 1
        assert game.game_over or moves_played == 200
        scores = game.get_scores()
        assert all(-89 <= s <= 20 for s in scores)
