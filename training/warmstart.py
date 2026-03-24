"""Warm-start data generation using rollout-MCTS.

Plays games where all 4 seats use the proven rollout-MCTS (no neural
network).  The resulting training examples give the neural network a
strong starting point before self-play begins.

Rollout-MCTS is CPU-only, so workers need no GPU access.
"""

from __future__ import annotations

import random
import time
from typing import List, Tuple

import torch.multiprocessing as mp

from blokus.board import NUM_PLAYERS
from blokus.game import Game
from mcts.search import MCTS
from nn.encoding import normalize_scores_for_nn
from .config import TrainingConfig
from .self_play import TrainingExample


# ------------------------------------------------------------------ #
# Single-game logic
# ------------------------------------------------------------------ #

def _play_one_warmstart_game(
    config: TrainingConfig,
    game_id: int,
    seed: int,
) -> Tuple[int, List[int], int, int, float, List[TrainingExample]]:
    """Play one full game with rollout-MCTS, collecting training examples."""
    t0 = time.perf_counter()

    mcts = MCTS(
        num_simulations=config.warmstart_simulations,
        exploration=1.41,
        rollout_rng=random.Random(seed),
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
        move, action_probs = mcts.search_for_training(game, temperature)

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
# Worker
# ------------------------------------------------------------------ #

def _worker_fn(
    config: TrainingConfig,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    """Worker: play rollout-MCTS games from the task queue."""
    while True:
        item = task_queue.get()
        if item is None:
            break
        game_id, seed = item
        result = _play_one_warmstart_game(config, game_id, seed)
        result_queue.put(result)


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

def run_warmstart(
    config: TrainingConfig,
    logger=None,
) -> List[TrainingExample]:
    """Generate training data from rollout-MCTS games.

    Uses multiprocessing to parallelise across CPU cores.
    """
    num_games = config.warmstart_games
    num_workers = min(config.num_workers, num_games)

    if num_workers <= 1:
        return _run_sequential(config, num_games, logger)
    return _run_parallel(config, num_games, num_workers, logger)


def _run_parallel(
    config: TrainingConfig,
    num_games: int,
    num_workers: int,
    logger,
) -> List[TrainingExample]:
    # Rollout-MCTS is CPU-only, so fork is safe and avoids spawn overhead
    ctx = mp.get_context("fork")
    task_q: mp.Queue = ctx.Queue()
    result_q: mp.Queue = ctx.Queue()

    for g in range(num_games):
        task_q.put((g, g * 7919))  # deterministic but varied seeds
    for _ in range(num_workers):
        task_q.put(None)

    workers = []
    for _ in range(num_workers):
        p = ctx.Process(target=_worker_fn, args=(config, task_q, result_q))
        p.start()
        workers.append(p)

    all_examples: List[TrainingExample] = []
    for _ in range(num_games):
        game_id, scores, winner, n_moves, elapsed, examples = result_q.get()
        all_examples.extend(examples)
        if logger is not None:
            logger.log_self_play_game(
                iteration=0, game_id=game_id,
                scores=scores, winner=winner,
                num_moves=n_moves, duration_s=elapsed,
            )
        print(
            f"    warmstart {len(all_examples)//60:>3}g  "
            f"moves={n_moves:>3}  scores={scores}  ({elapsed:.1f}s)"
        )

    for p in workers:
        p.join()

    return all_examples


def _run_sequential(
    config: TrainingConfig,
    num_games: int,
    logger,
) -> List[TrainingExample]:
    all_examples: List[TrainingExample] = []
    for g in range(num_games):
        result = _play_one_warmstart_game(config, g, g * 7919)
        game_id, scores, winner, n_moves, elapsed, examples = result
        all_examples.extend(examples)
        if logger is not None:
            logger.log_self_play_game(
                iteration=0, game_id=game_id,
                scores=scores, winner=winner,
                num_moves=n_moves, duration_s=elapsed,
            )
        print(
            f"    warmstart {g+1}/{num_games}  "
            f"moves={n_moves:>3}  scores={scores}  ({elapsed:.1f}s)"
        )
    return all_examples
