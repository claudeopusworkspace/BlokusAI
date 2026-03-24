"""Main training loop: self-play → train → evaluate → repeat.

Run with::

    python -m training.pipeline [--iterations N] [--games N] [--sims N]
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from torch.amp import GradScaler
from torch.optim import Adam

from blokus.board import NUM_PLAYERS
from blokus.game import Game
from blokus.players import RandomPlayer
from mcts.neural_mcts import NeuralMCTS
from nn.encoding import normalize_scores_for_nn, nn_value_to_mcts
from nn.model import BlokusNet
from .config import TrainingConfig
from .logger import TrainingLogger
from .self_play import run_self_play
from .trainer import ReplayBuffer, train_epoch


# ------------------------------------------------------------------ #
# Evaluation helper
# ------------------------------------------------------------------ #

def evaluate_vs_random(
    network: BlokusNet,
    config: TrainingConfig,
) -> tuple[int, float]:
    """Play *config.eval_games* against random players.

    The neural player sits at seat 0; seats 1–3 are random.
    Returns ``(wins, avg_score)``.
    """
    wins = 0
    total_score = 0

    for g in range(config.eval_games):
        mcts = NeuralMCTS(
            network=network,
            num_simulations=config.eval_mcts_sims,
            c_puct=config.c_puct,
        )
        game = Game()
        randoms = [RandomPlayer(seed=g * 10 + i) for i in range(1, NUM_PLAYERS)]

        while not game.game_over:
            p = game.current_player
            if p == 0:
                move, _ = mcts.search(game, temperature=0.0, add_noise=False)
            else:
                move = randoms[p - 1].choose_move(game)
            game.apply_move(move)

        scores = game.get_scores()
        if scores[0] == max(scores):
            wins += 1
        total_score += scores[0]

    return wins, total_score / config.eval_games


# ------------------------------------------------------------------ #
# Main loop
# ------------------------------------------------------------------ #

def run_training(config: TrainingConfig | None = None) -> None:
    if config is None:
        config = TrainingConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # --- set up ---
    network = BlokusNet(
        num_filters=config.num_filters,
        num_blocks=config.num_res_blocks,
        policy_channels=config.policy_channels,
        value_hidden=config.value_hidden,
    ).to(device)

    total_params = sum(p.numel() for p in network.parameters())
    print(f"Network: {total_params:,} parameters\n")

    optimizer = Adam(
        network.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = GradScaler("cuda") if device.type == "cuda" else None
    buffer = ReplayBuffer(config.replay_buffer_size)
    logger = TrainingLogger(config.log_dir)

    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- training loop ---
    for iteration in range(1, config.iterations + 1):
        print(f"{'='*60}")
        print(f"  Iteration {iteration}/{config.iterations}")
        print(f"{'='*60}")

        # 1) Self-play
        print("\n  [self-play]")
        t0 = time.perf_counter()
        examples = run_self_play(network, config, iteration, logger)
        self_play_time = time.perf_counter() - t0
        buffer.extend(examples)
        print(
            f"    → {len(examples)} examples "
            f"(buffer: {len(buffer)})  "
            f"({self_play_time:.1f}s)"
        )

        # 2) Train
        print("\n  [training]")
        t0 = time.perf_counter()
        epoch_policy_losses = []
        epoch_value_losses = []
        for epoch in range(config.epochs_per_iteration):
            pl, vl = train_epoch(
                network, optimizer, buffer, config, iteration, logger,
                scaler=scaler,
            )
            epoch_policy_losses.append(pl)
            epoch_value_losses.append(vl)
            print(
                f"    epoch {epoch+1}/{config.epochs_per_iteration}  "
                f"policy={pl:.4f}  value={vl:.4f}"
            )
        train_time = time.perf_counter() - t0
        avg_pl = sum(epoch_policy_losses) / len(epoch_policy_losses)
        avg_vl = sum(epoch_value_losses) / len(epoch_value_losses)

        # 3) Evaluate (periodically)
        eval_wr: float | None = None
        if iteration % config.eval_interval == 0 or iteration == 1:
            print("\n  [evaluation vs random]")
            t0 = time.perf_counter()
            wins, avg_score = evaluate_vs_random(network, config)
            eval_time = time.perf_counter() - t0
            eval_wr = wins / config.eval_games
            logger.log_evaluation(
                iteration=iteration,
                opponent="random",
                games=config.eval_games,
                wins=wins,
                avg_score=avg_score,
            )
            print(
                f"    wins={wins}/{config.eval_games} "
                f"({eval_wr:.0%})  "
                f"avg_score={avg_score:.1f}  "
                f"({eval_time:.1f}s)"
            )

        # 4) Log iteration summary
        logger.log_iteration(
            iteration=iteration,
            avg_policy_loss=avg_pl,
            avg_value_loss=avg_vl,
            buffer_size=len(buffer),
            self_play_time_s=self_play_time,
            training_time_s=train_time,
            eval_win_rate=eval_wr,
        )

        # 5) Checkpoint
        ckpt_path = ckpt_dir / f"model_iter_{iteration:04d}.pt"
        torch.save({
            "iteration": iteration,
            "model_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        }, ckpt_path)
        print(f"\n  checkpoint → {ckpt_path}")

    logger.close()
    print("\n✓ Training complete.")


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="BlokusAI training pipeline")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--games", type=int, default=20,
                        help="Self-play games per iteration")
    parser.add_argument("--sims", type=int, default=100,
                        help="MCTS simulations per move")
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--filters", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    config = TrainingConfig(
        iterations=args.iterations,
        games_per_iteration=args.games,
        num_simulations=args.sims,
        eval_games=args.eval_games,
        eval_interval=args.eval_interval,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_filters=args.filters,
        num_res_blocks=args.blocks,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
    )
    run_training(config)


if __name__ == "__main__":
    main()
