"""Neural MCTS with PUCT selection for 4-player Blokus.

Replaces random rollouts with neural-network evaluation.  Selection uses
PUCT (Predictor + Upper Confidence bounds applied to Trees) where the
network's policy output supplies move prior probabilities.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from blokus.board import Move, NUM_PLAYERS
from blokus.game import Game
from nn.encoding import (
    ACTION_SPACE_SIZE,
    encode_action,
    encode_legal_mask,
    normalize_scores_for_nn,
    nn_value_to_mcts,
)
from nn.model import BlokusNet


# ------------------------------------------------------------------ #
# Tree node
# ------------------------------------------------------------------ #

class NeuralMCTSNode:
    """Node storing per-player values and an NN-derived prior."""

    __slots__ = (
        "parent", "move", "player", "children",
        "prior", "visits", "value_sum", "is_terminal", "is_expanded",
    )

    def __init__(
        self,
        parent: Optional[NeuralMCTSNode],
        move: Optional[Move],
        player: int,
        prior: float = 0.0,
        is_terminal: bool = False,
    ) -> None:
        self.parent = parent
        self.move = move
        self.player = player
        self.children: Dict[Move, NeuralMCTSNode] = {}
        self.prior = prior
        self.visits: int = 0
        self.value_sum: np.ndarray = np.zeros(NUM_PLAYERS, dtype=np.float64)
        self.is_terminal = is_terminal
        self.is_expanded = False

    def mean_value(self, player: int) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum[player] / self.visits

    def puct_score(self, child: NeuralMCTSNode, c_puct: float) -> float:
        """PUCT score from **this** node's player's perspective."""
        q = child.mean_value(self.player)
        u = c_puct * child.prior * math.sqrt(self.visits) / (1 + child.visits)
        return q + u

    def best_child(self, c_puct: float) -> NeuralMCTSNode:
        return max(self.children.values(),
                   key=lambda c: self.puct_score(c, c_puct))


# ------------------------------------------------------------------ #
# Search driver
# ------------------------------------------------------------------ #

class NeuralMCTS:
    """MCTS guided by a BlokusNet policy + value network.

    Parameters
    ----------
    network : BlokusNet
        Trained (or initialising) network.
    num_simulations : int
        Tree simulations per ``search()`` call.
    c_puct : float
        PUCT exploration constant.
    dirichlet_alpha, dirichlet_epsilon : float
        Dirichlet noise mixed into root priors for exploration.
    """

    def __init__(
        self,
        network: BlokusNet,
        num_simulations: int = 100,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ) -> None:
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.device = next(network.parameters()).device

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def search(
        self,
        game: Game,
        temperature: float = 1.0,
        add_noise: bool = True,
    ) -> Tuple[Move, Dict[int, float]]:
        """Run MCTS and return ``(move, {action_idx: probability})``.

        The probability dict is the MCTS visit-count policy used as the
        training target for the policy head.
        """
        root = NeuralMCTSNode(
            parent=None, move=None,
            player=game.current_player,
            is_terminal=game.game_over,
        )

        # Expand root with NN evaluation
        self._evaluate_and_expand(root, game)

        # Dirichlet noise for exploration
        if add_noise and root.children:
            self._add_dirichlet_noise(root)

        # Simulations
        for _ in range(self.num_simulations):
            self._simulate(root, game)

        return self._select_move(root, temperature)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _evaluate_and_expand(
        self, node: NeuralMCTSNode, game: Game,
    ) -> np.ndarray:
        """NN-evaluate the position, create children with priors.

        Returns value array of shape ``(4,)`` in ``[0, 1]``.
        """
        if game.game_over:
            node.is_terminal = True
            node.is_expanded = True
            scores = game.get_scores()
            return nn_value_to_mcts(normalize_scores_for_nn(scores))

        # Forward pass
        state = torch.from_numpy(game.get_state_planes()).unsqueeze(0).to(self.device)
        self.network.eval()
        policy_logits, value = self.network(state)

        # Masked softmax over legal moves
        legal_moves = game.get_legal_moves()
        legal_mask = encode_legal_mask(legal_moves)

        logits = policy_logits[0].cpu().numpy()
        logits[~legal_mask] = -1e9
        shifted = logits - logits.max()
        exp_l = np.exp(shifted)
        probs = exp_l / exp_l.sum()

        # Create children
        for move in legal_moves:
            aidx = encode_action(move)
            child = NeuralMCTSNode(
                parent=node, move=move,
                player=-1,        # set on first traversal
                prior=float(probs[aidx]),
            )
            node.children[move] = child

        node.is_expanded = True
        return nn_value_to_mcts(value[0].cpu().numpy())

    def _add_dirichlet_noise(self, node: NeuralMCTSNode) -> None:
        children = list(node.children.values())
        noise = np.random.dirichlet(
            [self.dirichlet_alpha] * len(children)
        )
        eps = self.dirichlet_epsilon
        for child, eta in zip(children, noise):
            child.prior = (1 - eps) * child.prior + eps * eta

    def _simulate(self, root: NeuralMCTSNode, root_game: Game) -> None:
        node = root
        state = root_game.copy()

        # --- selection ---
        while node.is_expanded and not node.is_terminal and node.children:
            node = node.best_child(self.c_puct)
            state.apply_move(node.move)
            if node.player == -1:
                node.player = state.current_player

        # --- leaf evaluation ---
        if node.is_terminal:
            value = nn_value_to_mcts(
                normalize_scores_for_nn(state.get_scores())
            )
        else:
            value = self._evaluate_and_expand(node, state)

        # --- back-propagation ---
        while node is not None:
            node.visits += 1
            node.value_sum += value
            node = node.parent

    def _select_move(
        self,
        root: NeuralMCTSNode,
        temperature: float,
    ) -> Tuple[Move, Dict[int, float]]:
        moves: List[Move] = []
        visits: List[int] = []
        for move, child in root.children.items():
            moves.append(move)
            visits.append(child.visits)

        v = np.array(visits, dtype=np.float64)

        if temperature <= 0:
            idx = int(np.argmax(v))
            probs = np.zeros_like(v)
            probs[idx] = 1.0
        else:
            v_temp = np.power(v, 1.0 / temperature)
            probs = v_temp / v_temp.sum()
            idx = int(np.random.choice(len(moves), p=probs))

        action_probs: Dict[int, float] = {}
        for move, p in zip(moves, probs):
            action_probs[encode_action(move)] = float(p)

        return moves[idx], action_probs
