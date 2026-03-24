"""Tests for board state, legal moves, and game flow."""

import random

import pytest

from blokus.board import BoardState, Move, BOARD_SIZE, NUM_PLAYERS, STARTING_CORNERS
from blokus.game import Game
from blokus.pieces import PIECES, PIECE_NAMES


# ------------------------------------------------------------------ #
# Board basics
# ------------------------------------------------------------------ #


class TestBoardInit:
    def test_grid_shape(self):
        b = BoardState()
        assert b.grid.shape == (BOARD_SIZE, BOARD_SIZE)

    def test_grid_empty(self):
        b = BoardState()
        assert (b.grid == 0).all()

    def test_initial_corners(self):
        b = BoardState()
        for p in range(NUM_PLAYERS):
            assert b.corners[p] == {STARTING_CORNERS[p]}

    def test_all_pieces_remaining(self):
        b = BoardState()
        for p in range(NUM_PLAYERS):
            assert b.remaining_pieces[p] == set(PIECE_NAMES)


# ------------------------------------------------------------------ #
# First-move placement
# ------------------------------------------------------------------ #


class TestFirstMove:
    def test_first_move_must_cover_starting_corner(self):
        """Player 0's first piece must include cell (0, 0)."""
        b = BoardState()
        moves = b.get_legal_moves(0)
        assert len(moves) > 0
        for m in moves:
            piece = PIECES[m.piece_name]
            ori = piece.orientations[m.orientation_idx]
            cells = {(m.row + dr, m.col + dc) for dr, dc in ori.cells}
            assert STARTING_CORNERS[0] in cells, (
                f"{m} does not cover starting corner {STARTING_CORNERS[0]}"
            )

    def test_monomino_first_move(self):
        """The monomino has exactly one legal placement at the starting corner."""
        b = BoardState()
        moves = b.get_legal_moves(0)
        mono_moves = [m for m in moves if m.piece_name == "I1"]
        assert len(mono_moves) == 1
        assert mono_moves[0].row == 0
        assert mono_moves[0].col == 0


# ------------------------------------------------------------------ #
# Placement and constraints
# ------------------------------------------------------------------ #


class TestPlacement:
    def _place_monomino(self, board: BoardState, player: int) -> None:
        """Helper: place the I1 piece at the player's starting corner."""
        sr, sc = STARTING_CORNERS[player]
        board.place_piece(player, "I1", 0, sr, sc)

    def test_place_updates_grid(self):
        b = BoardState()
        self._place_monomino(b, 0)
        assert b.grid[0, 0] == 1

    def test_place_removes_from_remaining(self):
        b = BoardState()
        self._place_monomino(b, 0)
        assert "I1" not in b.remaining_pieces[0]

    def test_place_adds_corners(self):
        b = BoardState()
        self._place_monomino(b, 0)
        # After placing at (0,0), diagonal (1,1) should be a corner
        assert (1, 1) in b.corners[0]
        # Edge cells should NOT be corners
        assert (0, 1) not in b.corners[0]
        assert (1, 0) not in b.corners[0]

    def test_place_adds_edges(self):
        b = BoardState()
        self._place_monomino(b, 0)
        assert (0, 1) in b.edges[0]
        assert (1, 0) in b.edges[0]

    def test_second_piece_must_touch_corner(self):
        """After placing I1 at (0,0), next piece must touch a diagonal."""
        b = BoardState()
        self._place_monomino(b, 0)
        moves = b.get_legal_moves(0)
        for m in moves:
            piece = PIECES[m.piece_name]
            ori = piece.orientations[m.orientation_idx]
            cells = {(m.row + dr, m.col + dc) for dr, dc in ori.cells}
            # At least one cell must be in the player's corners
            assert cells & b.corners[0], f"{m} doesn't touch a corner"

    def test_no_edge_adjacency_to_own_color(self):
        """No piece cell may be edge-adjacent to the same player's existing piece."""
        b = BoardState()
        self._place_monomino(b, 0)
        moves = b.get_legal_moves(0)
        for m in moves:
            piece = PIECES[m.piece_name]
            ori = piece.orientations[m.orientation_idx]
            for dr, dc in ori.cells:
                r, c = m.row + dr, m.col + dc
                assert (r, c) not in b.edges[0], (
                    f"{m}: cell ({r},{c}) is edge-adjacent to own color"
                )


# ------------------------------------------------------------------ #
# Copy
# ------------------------------------------------------------------ #


class TestBoardCopy:
    def test_copy_is_independent(self):
        b = BoardState()
        b.place_piece(0, "I1", 0, 0, 0)
        b2 = b.copy()
        b2.place_piece(1, "I1", 0, 0, 19)
        # Original should be unchanged
        assert b.grid[0, 19] == 0
        assert "I1" in b.remaining_pieces[1]


# ------------------------------------------------------------------ #
# Game flow
# ------------------------------------------------------------------ #


class TestGameFlow:
    def test_initial_state(self):
        g = Game()
        assert g.current_player == 0
        assert not g.game_over
        assert len(g.move_history) == 0

    def test_apply_move_advances_player(self):
        g = Game()
        moves = g.get_legal_moves()
        g.apply_move(moves[0])
        assert g.current_player == 1

    def test_random_game_terminates(self):
        """A full random game should terminate without error."""
        rng = random.Random(123)
        g = Game()
        turns = 0
        max_turns = 500  # safety valve
        while not g.game_over and turns < max_turns:
            moves = g.get_legal_moves()
            if moves:
                g.apply_move(rng.choice(moves))
            else:
                g.skip_turn()
            turns += 1
        assert g.game_over
        assert turns < max_turns

    def test_scores_are_valid(self):
        """Run a random game and verify scores are in expected range."""
        rng = random.Random(456)
        g = Game()
        while not g.game_over:
            moves = g.get_legal_moves()
            if moves:
                g.apply_move(rng.choice(moves))
            else:
                g.skip_turn()
        scores = g.get_scores()
        for s in scores:
            # Worst case: -89 (no pieces placed). Best case: +20.
            assert -89 <= s <= 20

    def test_winner_exists(self):
        rng = random.Random(789)
        g = Game()
        while not g.game_over:
            moves = g.get_legal_moves()
            if moves:
                g.apply_move(rng.choice(moves))
            else:
                g.skip_turn()
        winner = g.get_winner()
        assert winner is not None
        assert 0 <= winner < NUM_PLAYERS


# ------------------------------------------------------------------ #
# State planes (for NN)
# ------------------------------------------------------------------ #


class TestStatePlanes:
    def test_shape(self):
        g = Game()
        planes = g.get_state_planes()
        assert planes.shape == (10, BOARD_SIZE, BOARD_SIZE)

    def test_initial_empty_plane(self):
        g = Game()
        planes = g.get_state_planes()
        # All cells empty at start
        assert planes[8].sum() == BOARD_SIZE * BOARD_SIZE

    def test_player_plane_after_move(self):
        g = Game()
        moves = g.get_legal_moves()
        g.apply_move(moves[0])
        planes = g.get_state_planes()
        # Player 0's plane should have some cells set
        assert planes[0].sum() > 0
