"""Board state management and legal move generation for Blokus.

The board is a 20x20 grid. Each cell is empty (0) or occupied by a player (1-4).
Per-player corner/edge tracking enables efficient legal move generation.
"""

from __future__ import annotations

from typing import List, Set, Tuple, Optional

import numpy as np

from .pieces import (
    PIECES,
    PIECE_NAMES,
    PieceData,
    Orientation,
    Cell,
    EDGE_DELTAS,
    CORNER_DELTAS,
)

BOARD_SIZE = 20
NUM_PLAYERS = 4

# Each player's mandatory starting corner (first piece must cover this cell)
STARTING_CORNERS: Tuple[Cell, ...] = (
    (0, 0),                         # Player 0 — top-left
    (0, BOARD_SIZE - 1),            # Player 1 — top-right
    (BOARD_SIZE - 1, BOARD_SIZE - 1),  # Player 2 — bottom-right
    (BOARD_SIZE - 1, 0),            # Player 3 — bottom-left
)


class Move:
    """A placement: piece + orientation + board position of the anchor cell."""

    __slots__ = ("piece_name", "orientation_idx", "row", "col")

    def __init__(self, piece_name: str, orientation_idx: int, row: int, col: int):
        self.piece_name = piece_name
        self.orientation_idx = orientation_idx
        self.row = row
        self.col = col

    def __repr__(self) -> str:
        return (
            f"Move({self.piece_name}, ori={self.orientation_idx}, "
            f"pos=({self.row},{self.col}))"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Move):
            return NotImplemented
        return (
            self.piece_name == other.piece_name
            and self.orientation_idx == other.orientation_idx
            and self.row == other.row
            and self.col == other.col
        )

    def __hash__(self) -> int:
        return hash((self.piece_name, self.orientation_idx, self.row, self.col))


class BoardState:
    """Mutable board state with incremental corner/edge tracking.

    Call :meth:`place_piece` to mutate, or :meth:`copy` for undo support.
    """

    def __init__(self) -> None:
        # 0 = empty, 1–4 = player id
        self.grid: np.ndarray = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)

        self.remaining_pieces: List[Set[str]] = [
            set(PIECE_NAMES) for _ in range(NUM_PLAYERS)
        ]

        # Corner candidates: cells where the player *may* anchor a new piece
        # (diagonally adjacent to own piece, not edge-adjacent to own piece,
        #  and currently empty).
        self.corners: List[Set[Cell]] = [
            {STARTING_CORNERS[p]} for p in range(NUM_PLAYERS)
        ]

        # Edge-adjacent cells for each player (placement forbidden for own color)
        self.edges: List[Set[Cell]] = [set() for _ in range(NUM_PLAYERS)]

        # Track whether each player has placed at least one piece
        self.has_placed: List[bool] = [False] * NUM_PLAYERS

    # ------------------------------------------------------------------ #
    # Deep copy
    # ------------------------------------------------------------------ #

    def copy(self) -> BoardState:
        new = BoardState.__new__(BoardState)
        new.grid = self.grid.copy()
        new.remaining_pieces = [s.copy() for s in self.remaining_pieces]
        new.corners = [s.copy() for s in self.corners]
        new.edges = [s.copy() for s in self.edges]
        new.has_placed = self.has_placed.copy()
        return new

    # ------------------------------------------------------------------ #
    # Placement validation
    # ------------------------------------------------------------------ #

    def is_valid_placement(
        self, player: int, orientation: Orientation, row: int, col: int
    ) -> bool:
        """Check whether placing *orientation* at (row, col) is legal for *player*."""
        touches_corner = False
        player_edges = self.edges[player]
        player_corners = self.corners[player]
        grid = self.grid

        for dr, dc in orientation.cells:
            r = row + dr
            c = col + dc
            # Bounds check
            if r < 0 or r >= BOARD_SIZE or c < 0 or c >= BOARD_SIZE:
                return False
            # Must be empty
            if grid[r, c] != 0:
                return False
            # Must not be edge-adjacent to own color
            if (r, c) in player_edges:
                return False
            # Does this cell sit on a corner candidate?
            if (r, c) in player_corners:
                touches_corner = True

        return touches_corner

    # ------------------------------------------------------------------ #
    # Apply a move (mutates in place)
    # ------------------------------------------------------------------ #

    def place_piece(
        self, player: int, piece_name: str, orientation_idx: int, row: int, col: int
    ) -> None:
        """Place a piece on the board.  Assumes the move has been validated."""
        piece = PIECES[piece_name]
        orientation = piece.orientations[orientation_idx]
        pid = player + 1  # grid stores 1–4

        placed: List[Cell] = []
        for dr, dc in orientation.cells:
            r, c = row + dr, col + dc
            self.grid[r, c] = pid
            placed.append((r, c))

        self.remaining_pieces[player].discard(piece_name)
        self.has_placed[player] = True

        # --- Update corner / edge bookkeeping ---
        player_corners = self.corners[player]
        player_edges = self.edges[player]

        for r, c in placed:
            # Occupied cells can't be corner candidates for anyone
            for p in range(NUM_PLAYERS):
                self.corners[p].discard((r, c))

            # Edge neighbors become forbidden for this player
            for dr2, dc2 in EDGE_DELTAS:
                nr, nc = r + dr2, c + dc2
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    player_edges.add((nr, nc))
                    player_corners.discard((nr, nc))

            # Diagonal neighbors become new corner candidates
            for dr2, dc2 in CORNER_DELTAS:
                nr, nc = r + dr2, c + dc2
                if (
                    0 <= nr < BOARD_SIZE
                    and 0 <= nc < BOARD_SIZE
                    and self.grid[nr, nc] == 0
                    and (nr, nc) not in player_edges
                ):
                    player_corners.add((nr, nc))

    # ------------------------------------------------------------------ #
    # Legal move generation
    # ------------------------------------------------------------------ #

    def get_legal_moves(self, player: int) -> List[Move]:
        """Return every legal move for *player* in the current state."""
        moves_seen: set[Tuple[str, int, int, int]] = set()
        moves: List[Move] = []

        remaining = self.remaining_pieces[player]
        corners = self.corners[player]

        if not corners or not remaining:
            return moves

        for piece_name in remaining:
            piece = PIECES[piece_name]
            for ori_idx, orientation in enumerate(piece.orientations):
                for cr, cc in corners:
                    # Try anchoring so that each cell of the piece lands on
                    # this corner candidate.
                    for dr, dc in orientation.cells:
                        anchor_r = cr - dr
                        anchor_c = cc - dc
                        key = (piece_name, ori_idx, anchor_r, anchor_c)
                        if key in moves_seen:
                            continue
                        if self.is_valid_placement(
                            player, orientation, anchor_r, anchor_c
                        ):
                            moves_seen.add(key)
                            moves.append(
                                Move(piece_name, ori_idx, anchor_r, anchor_c)
                            )

        return moves

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #

    def remaining_squares(self, player: int) -> int:
        """Total squares in pieces the player has NOT yet placed."""
        return sum(
            PIECES[name].size for name in self.remaining_pieces[player]
        )
