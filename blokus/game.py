"""High-level game flow for Blokus.

Manages turn order, pass detection, scoring, and game-over conditions.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .board import BoardState, Move, NUM_PLAYERS
from .pieces import PIECES


class Game:
    """A complete Blokus game session."""

    def __init__(self) -> None:
        self.board = BoardState()
        self.current_player: int = 0
        self.game_over: bool = False
        self.move_history: List[Tuple[int, Optional[Move]]] = []
        # Track last piece placed per player (for monomino bonus)
        self.last_piece: List[Optional[str]] = [None] * NUM_PLAYERS
        # Cache of legal moves for the current player (invalidated on move)
        self._legal_moves: Optional[List[Move]] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_legal_moves(self) -> List[Move]:
        """Legal moves for the current player (cached until a move is made)."""
        if self.game_over:
            return []
        if self._legal_moves is None:
            self._legal_moves = self.board.get_legal_moves(self.current_player)
        return self._legal_moves

    def apply_move(self, move: Move) -> None:
        """Apply *move* for the current player and advance the turn."""
        player = self.current_player
        self.board.place_piece(
            player, move.piece_name, move.orientation_idx, move.row, move.col
        )
        self.last_piece[player] = move.piece_name
        self.move_history.append((player, move))
        self._legal_moves = None
        self._advance()

    def skip_turn(self) -> None:
        """Record that the current player cannot move, then advance."""
        self.move_history.append((self.current_player, None))
        self._legal_moves = None
        self._advance()

    def get_scores(self) -> List[int]:
        """Score for each player.

        * ``-1`` per remaining square.
        * ``+15`` bonus if all 21 pieces placed.
        * ``+5`` extra if the monomino (I1) was the last piece placed.
        """
        scores: List[int] = []
        for p in range(NUM_PLAYERS):
            remaining = self.board.remaining_pieces[p]
            if not remaining:
                bonus = 15
                if self.last_piece[p] == "I1":
                    bonus += 5
                scores.append(bonus)
            else:
                scores.append(-self.board.remaining_squares(p))
        return scores

    def get_winner(self) -> Optional[int]:
        """Return the player index with the highest score, or ``None``."""
        if not self.game_over:
            return None
        scores = self.get_scores()
        return int(scores.index(max(scores)))

    def get_rankings(self) -> List[int]:
        """Return player indices sorted by score (best first)."""
        scores = self.get_scores()
        return sorted(range(NUM_PLAYERS), key=lambda p: scores[p], reverse=True)

    # ------------------------------------------------------------------ #
    # Copy (for MCTS simulations)
    # ------------------------------------------------------------------ #

    def copy(self) -> Game:
        """Deep-copy the game state. Move history is shared (copy-on-write)."""
        new = Game.__new__(Game)
        new.board = self.board.copy()
        new.current_player = self.current_player
        new.game_over = self.game_over
        new.move_history = list(self.move_history)
        new.last_piece = list(self.last_piece)
        new._legal_moves = None
        return new

    # ------------------------------------------------------------------ #
    # State for neural network input
    # ------------------------------------------------------------------ #

    def get_state_planes(self) -> "np.ndarray":  # noqa: F821
        """Return a tensor of binary feature planes for the NN.

        Shape: ``(C, 20, 20)`` where the channels are:
          0–3 : binary mask of each player's pieces
          4–7 : binary mask of each player's corner candidates
          8   : binary mask of empty cells
          9   : whose turn it is (constant plane, value = current_player / 3)
        """
        import numpy as np
        from .board import BOARD_SIZE

        planes = np.zeros((10, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        for p in range(NUM_PLAYERS):
            planes[p] = (self.board.grid == (p + 1)).astype(np.float32)
            for r, c in self.board.corners[p]:
                planes[4 + p, r, c] = 1.0
        planes[8] = (self.board.grid == 0).astype(np.float32)
        planes[9] = self.current_player / 3.0
        return planes

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _advance(self) -> None:
        """Move to the next player who has at least one legal move."""
        start = self.current_player
        for i in range(1, NUM_PLAYERS + 1):
            candidate = (start + i) % NUM_PLAYERS
            if self.board.get_legal_moves(candidate):
                self.current_player = candidate
                self._legal_moves = None
                return
        # Nobody can move — game over
        self.game_over = True
