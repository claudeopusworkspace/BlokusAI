"""4-player Monte Carlo Tree Search with UCB1 selection.

Each tree node tracks per-player value estimates so that each player
maximises their own expected outcome during selection.
"""

from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from blokus.board import Move, NUM_PLAYERS
from blokus.game import Game
from blokus.pieces import PIECES
from nn.encoding import encode_action

# Blokus score range (for normalization to [0, 1])
_MIN_SCORE = -89
_MAX_SCORE = 20
_SCORE_RANGE = _MAX_SCORE - _MIN_SCORE  # 109


def _normalize_scores(scores: List[int]) -> np.ndarray:
    """Map raw Blokus scores to [0, 1]."""
    return np.array([(s - _MIN_SCORE) / _SCORE_RANGE for s in scores],
                    dtype=np.float64)


# ------------------------------------------------------------------ #
# Tree node
# ------------------------------------------------------------------ #

class MCTSNode:
    """A node in the MCTS game tree.

    Attributes:
        parent:          Parent node (``None`` for the root).
        move:            The :class:`Move` that led here from the parent.
        player:          Index of the player whose turn it is at this node.
        children:        Expanded children keyed by :class:`Move`.
        untried_moves:   Legal moves not yet expanded (``None`` = uninitialised).
        visits:          Number of times this node was visited (N).
        value_sum:       Running total of normalised rewards per player (W).
        is_terminal:     Whether the game is over at this node.
    """

    __slots__ = (
        "parent", "move", "player", "children", "untried_moves",
        "visits", "value_sum", "is_terminal",
    )

    def __init__(
        self,
        parent: Optional[MCTSNode],
        move: Optional[Move],
        player: int,
        is_terminal: bool = False,
    ) -> None:
        self.parent = parent
        self.move = move
        self.player = player
        self.children: Dict[Move, MCTSNode] = {}
        self.untried_moves: Optional[List[Move]] = None
        self.visits: int = 0
        self.value_sum: np.ndarray = np.zeros(NUM_PLAYERS, dtype=np.float64)
        self.is_terminal = is_terminal

    @property
    def is_fully_expanded(self) -> bool:
        return self.untried_moves is not None and len(self.untried_moves) == 0

    def mean_value(self, player: int) -> float:
        """Average normalised value for *player* at this node."""
        if self.visits == 0:
            return 0.0
        return self.value_sum[player] / self.visits

    def ucb(self, child: "MCTSNode", exploration: float) -> float:
        """UCB1 score for *child* from **this** node's player's perspective."""
        exploitation = child.value_sum[self.player] / child.visits
        explore = exploration * math.sqrt(math.log(self.visits) / child.visits)
        return exploitation + explore

    def best_child(self, exploration: float) -> "MCTSNode":
        """Select the child with the highest UCB1 score."""
        return max(
            self.children.values(),
            key=lambda c: self.ucb(c, exploration),
        )

    def __repr__(self) -> str:
        return (
            f"MCTSNode(player={self.player}, visits={self.visits}, "
            f"children={len(self.children)}, "
            f"untried={len(self.untried_moves) if self.untried_moves is not None else '?'})"
        )


# ------------------------------------------------------------------ #
# Search driver
# ------------------------------------------------------------------ #

class MCTS:
    """Monte Carlo Tree Search for 4-player Blokus.

    Parameters:
        num_simulations:  Number of playouts per :meth:`search` call.
        exploration:      UCB1 exploration constant *c*.
        rollout_rng:      Optional :class:`random.Random` for determinism.
    """

    def __init__(
        self,
        num_simulations: int = 200,
        exploration: float = 1.41,
        rollout_rng: Optional[random.Random] = None,
    ) -> None:
        self.num_simulations = num_simulations
        self.exploration = exploration
        self.rng = rollout_rng or random.Random()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def search(
        self,
        game: Game,
        temperature: float = 0.0,
    ) -> Tuple[Move, dict]:
        """Run MCTS and return ``(chosen_move, stats)``.

        *temperature* controls move selection from the root visit counts:
        ``0`` = pick most-visited (greedy), ``>0`` = sample proportionally.
        """
        t0 = time.perf_counter()

        root = MCTSNode(
            parent=None, move=None,
            player=game.current_player,
            is_terminal=game.game_over,
        )
        root.untried_moves = list(game.get_legal_moves())

        for _ in range(self.num_simulations):
            self._simulate(root, game)

        move, probs = self._select_move(root, temperature)

        elapsed = time.perf_counter() - t0
        stats = {
            "simulations": self.num_simulations,
            "root_visits": root.visits,
            "children": len(root.children),
            "elapsed_s": elapsed,
            "sims_per_sec": self.num_simulations / elapsed if elapsed > 0 else 0,
            "best_value": max(
                c.mean_value(root.player) for c in root.children.values()
            ) if root.children else 0.0,
        }
        return move, stats

    def search_for_training(
        self,
        game: Game,
        temperature: float = 1.0,
    ) -> Tuple[Move, Dict[int, float]]:
        """Run MCTS and return ``(move, {action_idx: probability})``.

        Same tree search as :meth:`search`, but returns the visit-count
        policy in the format expected by :class:`TrainingExample`.
        """
        root = MCTSNode(
            parent=None, move=None,
            player=game.current_player,
            is_terminal=game.game_over,
        )
        root.untried_moves = list(game.get_legal_moves())

        for _ in range(self.num_simulations):
            self._simulate(root, game)

        move, probs = self._select_move(root, temperature)

        # Convert to {action_index: probability} dict
        action_probs: Dict[int, float] = {}
        moves_list = list(root.children.keys())
        for m, p in zip(moves_list, probs):
            action_probs[encode_action(m)] = float(p)

        return move, action_probs

    def get_move_probabilities(
        self,
        root: MCTSNode,
    ) -> Tuple[List[Move], np.ndarray]:
        """Return (moves, visit_count_array) for the root's children."""
        moves: List[Move] = []
        visits: List[int] = []
        for move, child in root.children.items():
            moves.append(move)
            visits.append(child.visits)
        return moves, np.array(visits, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Core MCTS loop
    # ------------------------------------------------------------------ #

    def _simulate(self, root: MCTSNode, root_game: Game) -> None:
        """One iteration: select → expand → rollout → backpropagate."""
        node = root
        state = root_game.copy()

        # --- Selection ---
        while node.is_fully_expanded and not node.is_terminal and node.children:
            node = node.best_child(self.exploration)
            state.apply_move(node.move)

        # --- Terminal node: just backprop ---
        if node.is_terminal:
            result = _normalize_scores(state.get_scores())
            self._backpropagate(node, result)
            return

        # --- Lazy initialisation of untried moves ---
        if node.untried_moves is None:
            if state.game_over:
                node.is_terminal = True
                result = _normalize_scores(state.get_scores())
                self._backpropagate(node, result)
                return
            node.untried_moves = list(state.get_legal_moves())

        # --- Expansion ---
        move = self.rng.choice(node.untried_moves)
        node.untried_moves.remove(move)

        state.apply_move(move)
        child = MCTSNode(
            parent=node,
            move=move,
            player=state.current_player,
            is_terminal=state.game_over,
        )
        node.children[move] = child
        node = child

        # --- Rollout ---
        if node.is_terminal:
            result = _normalize_scores(state.get_scores())
        else:
            result = self._rollout(state)

        # --- Backpropagation ---
        self._backpropagate(node, result)

    def _rollout(self, state: Game) -> np.ndarray:
        """Play random moves until the game ends; return normalised scores."""
        while not state.game_over:
            moves = state.get_legal_moves()
            state.apply_move(self.rng.choice(moves))
        return _normalize_scores(state.get_scores())

    @staticmethod
    def _backpropagate(node: MCTSNode, result: np.ndarray) -> None:
        """Walk back to the root, updating visit counts and value sums."""
        while node is not None:
            node.visits += 1
            node.value_sum += result
            node = node.parent

    # ------------------------------------------------------------------ #
    # Move selection
    # ------------------------------------------------------------------ #

    def _select_move(
        self, root: MCTSNode, temperature: float,
    ) -> Tuple[Move, np.ndarray]:
        """Choose a move from the root based on visit counts.

        Returns ``(move, probability_vector)`` over the root's children.
        """
        moves: List[Move] = []
        visits: List[int] = []
        for move, child in root.children.items():
            moves.append(move)
            visits.append(child.visits)

        v = np.array(visits, dtype=np.float64)

        if temperature <= 0:
            # Greedy: pick most-visited
            idx = int(np.argmax(v))
            probs = np.zeros_like(v)
            probs[idx] = 1.0
        else:
            # Sample proportionally to visit_count^(1/temp)
            log_v = np.log(v + 1e-8) / temperature
            log_v -= log_v.max()  # numerical stability
            probs = np.exp(log_v)
            probs /= probs.sum()
            idx = self.rng.choices(range(len(moves)), weights=probs)[0]

        return moves[idx], probs
