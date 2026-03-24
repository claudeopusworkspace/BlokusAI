"""Action-space encoding and score normalisation.

Maps ``(piece_name, orientation_idx, row, col)`` ↔ flat action index.

Layout
------
For each (piece, orientation) pair in canonical order there are
``BOARD_SIZE² = 400`` slots (one per board position).

    action_index = pair_index × 400 + row × 20 + col

Total action space: 91 pairs × 400 = **36 400**.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from blokus.board import BOARD_SIZE, Move
from blokus.pieces import PIECES, PIECE_NAMES

# ------------------------------------------------------------------ #
# Build canonical (piece, orientation) pair table
# ------------------------------------------------------------------ #

_PAIR_OFFSET: dict[str, int] = {}
_PAIR_LIST: list[tuple[str, int]] = []

_idx = 0
for _name in PIECE_NAMES:
    _PAIR_OFFSET[_name] = _idx
    for _oi in range(len(PIECES[_name].orientations)):
        _PAIR_LIST.append((_name, _oi))
        _idx += 1

NUM_PAIRS: int = _idx                                    # 91
BOARD_CELLS: int = BOARD_SIZE * BOARD_SIZE                # 400
ACTION_SPACE_SIZE: int = NUM_PAIRS * BOARD_CELLS          # 36 400

assert NUM_PAIRS == 91
assert ACTION_SPACE_SIZE == 36_400

# ------------------------------------------------------------------ #
# Encode / decode
# ------------------------------------------------------------------ #


def encode_action(move: Move) -> int:
    """Move → flat action index."""
    pair = _PAIR_OFFSET[move.piece_name] + move.orientation_idx
    return pair * BOARD_CELLS + move.row * BOARD_SIZE + move.col


def decode_action(action: int) -> Move:
    """Flat action index → Move."""
    pair, pos = divmod(action, BOARD_CELLS)
    row, col = divmod(pos, BOARD_SIZE)
    piece_name, ori_idx = _PAIR_LIST[pair]
    return Move(piece_name, ori_idx, row, col)


def encode_legal_mask(legal_moves: list[Move]) -> np.ndarray:
    """Boolean mask over the action space (True = legal)."""
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
    for m in legal_moves:
        mask[encode_action(m)] = True
    return mask


def encode_policy(move_probs: Dict[int, float]) -> np.ndarray:
    """``{action_index: prob}`` → dense float32 vector of size 36 400."""
    policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    for idx, prob in move_probs.items():
        policy[idx] = prob
    return policy


# ------------------------------------------------------------------ #
# Score normalisation
# ------------------------------------------------------------------ #

_MIN_SCORE = -89
_MAX_SCORE = 20
_SCORE_RANGE = _MAX_SCORE - _MIN_SCORE  # 109


def normalize_scores_for_nn(scores: List[int]) -> np.ndarray:
    """Raw Blokus scores → [-1, 1] (for NN value head / training target)."""
    return np.array(
        [(s - _MIN_SCORE) / (_SCORE_RANGE / 2) - 1.0 for s in scores],
        dtype=np.float32,
    )


def nn_value_to_mcts(value: np.ndarray) -> np.ndarray:
    """NN value in [-1, 1] → MCTS value in [0, 1]."""
    return (np.asarray(value, dtype=np.float64) + 1.0) / 2.0
