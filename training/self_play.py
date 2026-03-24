"""Self-play game generation using Neural MCTS.

Each game produces a list of :class:`TrainingExample` instances that pair
board states with MCTS-derived policy targets and final-outcome value
targets.

Self-play games run in parallel across CPU cores using
``torch.multiprocessing``, since the bottleneck is the Python game engine
(legal-move generation), not the GPU.  Each worker gets its own model
copy on the GPU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp

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


# ------------------------------------------------------------------ #
# Single-game logic (called from any context)
# ------------------------------------------------------------------ #

def _play_one_game(
    network: BlokusNet,
    config: TrainingConfig,
    game_id: int,
) -> Tuple[int, List[int], int, int, float, List[TrainingExample]]:
    """Play one self-play game.  Returns metadata + training examples."""
    t0 = time.perf_counter()

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
        temperature = (
            1.0 if move_count < config.temp_threshold
            else config.temp_final
        )
        move, action_probs = mcts.search(
            game, temperature=temperature, add_noise=True,
        )
        examples.append(TrainingExample(
            state=state_planes,
            policy=action_probs,
        ))
        game.apply_move(move)
        move_count += 1

    scores = game.get_scores()
    value_target = normalize_scores_for_nn(scores)
    for ex in examples:
        ex.value = value_target.copy()

    elapsed = time.perf_counter() - t0
    winner = game.get_winner()
    return (game_id, scores, winner if winner is not None else -1,
            len(examples), elapsed, examples)


# ------------------------------------------------------------------ #
# Worker process
# ------------------------------------------------------------------ #

def _worker_fn(
    state_dict: dict,
    device_str: str,
    config: TrainingConfig,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    """Worker: reconstruct model on GPU, then play games from the queue."""
    device = torch.device(device_str)
    network = BlokusNet(
        num_filters=config.num_filters,
        num_blocks=config.num_res_blocks,
        policy_channels=config.policy_channels,
        value_hidden=config.value_hidden,
    ).to(device)
    network.load_state_dict(state_dict)
    network.eval()

    while True:
        game_id = task_queue.get()
        if game_id is None:
            break
        result = _play_one_game(network, config, game_id)
        result_queue.put(result)


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

def run_self_play(
    network: BlokusNet,
    config: TrainingConfig,
    iteration: int,
    logger=None,
) -> List[TrainingExample]:
    """Play ``config.games_per_iteration`` games, optionally in parallel."""
    num_games = config.games_per_iteration
    num_workers = min(config.num_workers, num_games)

    if num_workers <= 1:
        return _run_sequential(network, config, iteration, logger)
    return _run_parallel(network, config, iteration, logger,
                         num_workers, num_games)


def _run_parallel(
    network: BlokusNet,
    config: TrainingConfig,
    iteration: int,
    logger,
    num_workers: int,
    num_games: int,
) -> List[TrainingExample]:
    """Multi-process self-play.  Each worker gets its own GPU model copy."""
    device = next(network.parameters()).device

    # Snapshot weights to CPU for pickling to workers
    cpu_state = {k: v.cpu() for k, v in network.state_dict().items()}

    ctx = mp.get_context("spawn")
    task_q: mp.Queue = ctx.Queue()
    result_q: mp.Queue = ctx.Queue()

    for g in range(num_games):
        task_q.put(g)
    for _ in range(num_workers):
        task_q.put(None)           # poison pills

    workers = []
    for _ in range(num_workers):
        p = ctx.Process(
            target=_worker_fn,
            args=(cpu_state, str(device), config, task_q, result_q),
        )
        p.start()
        workers.append(p)

    all_examples: List[TrainingExample] = []
    completed = 0
    for _ in range(num_games):
        game_id, scores, winner, n_moves, elapsed, examples = result_q.get()
        all_examples.extend(examples)
        completed += 1
        if logger is not None:
            logger.log_self_play_game(
                iteration=iteration, game_id=game_id,
                scores=scores, winner=winner,
                num_moves=n_moves, duration_s=elapsed,
            )
        print(
            f"    self-play {completed:>2}/{num_games}  "
            f"moves={n_moves:>3}  scores={scores}  ({elapsed:.1f}s)"
        )

    for p in workers:
        p.join()

    return all_examples


def _run_sequential(
    network: BlokusNet,
    config: TrainingConfig,
    iteration: int,
    logger,
) -> List[TrainingExample]:
    """Single-process fallback."""
    all_examples: List[TrainingExample] = []
    for g in range(config.games_per_iteration):
        result = _play_one_game(network, config, g)
        game_id, scores, winner, n_moves, elapsed, examples = result
        all_examples.extend(examples)
        if logger is not None:
            logger.log_self_play_game(
                iteration=iteration, game_id=game_id,
                scores=scores, winner=winner,
                num_moves=n_moves, duration_s=elapsed,
            )
        print(
            f"    self-play {g+1}/{config.games_per_iteration}  "
            f"moves={n_moves:>3}  scores={scores}  ({elapsed:.1f}s)"
        )
    return all_examples
