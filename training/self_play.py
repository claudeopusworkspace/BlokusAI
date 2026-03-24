"""Self-play game generation using Neural MCTS.

Each game produces a list of :class:`TrainingExample` instances that pair
board states with MCTS-derived policy targets and final-outcome value
targets.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from blokus.board import NUM_PLAYERS
from blokus.game import Game
from mcts.neural_mcts import NeuralMCTS
from nn.encoding import normalize_scores_for_nn
from nn.model import BlokusNet
from .config import TrainingConfig


@dataclass
class TrainingExample:
    """One position from a self-play game."""

    state: np.ndarray              # (10, 20, 20) float32
    policy: Dict[int, float]       # {action_idx: MCTS probability}
    value: np.ndarray = field(     # (4,) float32, in [-1, 1]
        default_factory=lambda: np.zeros(NUM_PLAYERS, dtype=np.float32)
    )


def play_self_play_game(
    network: BlokusNet,
    config: TrainingConfig,
) -> tuple[Game, List[TrainingExample]]:
    """Play one full self-play game, returning the game and training data.

    All four seats use the same network.  Temperature is 1.0 for the
    first ``config.temp_threshold`` moves (encouraging exploration), then
    drops to 0.0 (greedy).
    """
    mcts = NeuralMCTS(
        network=network,
        num_simulations=config.num_simulations,
        c_puct=config.c_puct,
        dirichlet_alpha=config.dirichlet_alpha,
        dirichlet_epsilon=config.dirichlet_epsilon,
    )

    game = Game()
    examples: List[TrainingExample] = []
    move_count = 0

    while not game.game_over:
        state_planes = game.get_state_planes()

        temperature = 1.0 if move_count < config.temp_threshold else 0.0
        move, action_probs = mcts.search(
            game, temperature=temperature, add_noise=True,
        )

        examples.append(TrainingExample(
            state=state_planes,
            policy=action_probs,
        ))

        game.apply_move(move)
        move_count += 1

    # Back-fill value targets with the actual game outcome
    value_target = normalize_scores_for_nn(game.get_scores())
    for ex in examples:
        ex.value = value_target.copy()

    return game, examples


def run_self_play(
    network: BlokusNet,
    config: TrainingConfig,
    iteration: int,
    logger=None,
) -> List[TrainingExample]:
    """Play ``config.games_per_iteration`` games and collect training data."""
    all_examples: List[TrainingExample] = []

    for g in range(config.games_per_iteration):
        t0 = time.perf_counter()
        game, examples = play_self_play_game(network, config)
        elapsed = time.perf_counter() - t0

        all_examples.extend(examples)
        scores = game.get_scores()
        winner = game.get_winner()

        if logger is not None:
            logger.log_self_play_game(
                iteration=iteration,
                game_id=g,
                scores=scores,
                winner=winner if winner is not None else -1,
                num_moves=len(examples),
                duration_s=elapsed,
            )

        print(
            f"    self-play {g+1}/{config.games_per_iteration}  "
            f"moves={len(examples):>3}  "
            f"scores={scores}  "
            f"({elapsed:.1f}s)"
        )

    return all_examples
