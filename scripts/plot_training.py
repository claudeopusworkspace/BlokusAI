#!/usr/bin/env python3
"""Generate training analysis plots from JSONL logs.

Usage::

    python scripts/plot_training.py [--log-dir logs] [--out-dir plots]

Produces:
    1. loss_curves.png        — policy / value / total loss per iteration
    2. win_rate.png           — win-rate vs random over training
    3. self_play_scores.png   — per-player score distribution over training
    4. game_length.png        — average game length (moves) over training
    5. training_summary.png   — 2×2 combined dashboard
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def plot_all(log_dir: str = "logs", out_dir: str = "plots") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required: pip install matplotlib")
        return

    log_path = Path(log_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    training = _load_jsonl(log_path / "training.jsonl")
    self_play = _load_jsonl(log_path / "self_play.jsonl")
    evaluation = _load_jsonl(log_path / "evaluation.jsonl")
    iterations = _load_jsonl(log_path / "iterations.jsonl")

    if not iterations:
        print("No iteration data found — nothing to plot.")
        return

    iters = [r["iteration"] for r in iterations]
    plt.style.use("seaborn-v0_8-whitegrid")

    # ------------------------------------------------------------------
    # 1. Loss curves
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, [r["avg_policy_loss"] for r in iterations],
            "o-", label="Policy loss", markersize=3)
    ax.plot(iters, [r["avg_value_loss"] for r in iterations],
            "s-", label="Value loss", markersize=3)
    ax.plot(iters,
            [r["avg_policy_loss"] + r["avg_value_loss"] for r in iterations],
            "^-", label="Total loss", markersize=3, alpha=0.6)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path / "loss_curves.png", dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path / 'loss_curves.png'}")

    # ------------------------------------------------------------------
    # 2. Win rate vs random
    # ------------------------------------------------------------------
    if evaluation:
        fig, ax = plt.subplots(figsize=(8, 5))
        ev_iters = [r["iteration"] for r in evaluation]
        ev_wr = [r["win_rate"] for r in evaluation]
        ax.plot(ev_iters, ev_wr, "o-", color="tab:green", markersize=5)
        ax.axhline(0.25, color="gray", linestyle="--", alpha=0.7,
                    label="Random baseline (25%)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Win Rate")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Win Rate vs Random Players")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path / "win_rate.png", dpi=150)
        plt.close(fig)
        print(f"  wrote {out_path / 'win_rate.png'}")

    # ------------------------------------------------------------------
    # 3. Self-play score distribution per iteration
    # ------------------------------------------------------------------
    if self_play:
        fig, ax = plt.subplots(figsize=(8, 5))

        sp_iters = sorted(set(r["iteration"] for r in self_play))
        avg_best = []
        avg_worst = []
        avg_mean = []
        for it in sp_iters:
            games = [r for r in self_play if r["iteration"] == it]
            bests = [max(g["scores"]) for g in games]
            worsts = [min(g["scores"]) for g in games]
            means = [np.mean(g["scores"]) for g in games]
            avg_best.append(np.mean(bests))
            avg_worst.append(np.mean(worsts))
            avg_mean.append(np.mean(means))

        ax.fill_between(sp_iters, avg_worst, avg_best,
                        alpha=0.2, color="tab:blue", label="Best–worst range")
        ax.plot(sp_iters, avg_mean, "o-", color="tab:blue",
                markersize=3, label="Mean score")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Score")
        ax.set_title("Self-Play Score Distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path / "self_play_scores.png", dpi=150)
        plt.close(fig)
        print(f"  wrote {out_path / 'self_play_scores.png'}")

    # ------------------------------------------------------------------
    # 4. Game length
    # ------------------------------------------------------------------
    if self_play:
        fig, ax = plt.subplots(figsize=(8, 5))

        avg_moves = []
        for it in sp_iters:
            games = [r for r in self_play if r["iteration"] == it]
            avg_moves.append(np.mean([g["num_moves"] for g in games]))

        ax.plot(sp_iters, avg_moves, "o-", color="tab:orange", markersize=3)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Moves per Game")
        ax.set_title("Average Game Length")
        fig.tight_layout()
        fig.savefig(out_path / "game_length.png", dpi=150)
        plt.close(fig)
        print(f"  wrote {out_path / 'game_length.png'}")

    # ------------------------------------------------------------------
    # 5. Combined dashboard
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    ax = axes[0, 0]
    ax.plot(iters, [r["avg_policy_loss"] for r in iterations],
            "o-", label="Policy", markersize=3)
    ax.plot(iters, [r["avg_value_loss"] for r in iterations],
            "s-", label="Value", markersize=3)
    ax.set_title("Training Loss")
    ax.set_xlabel("Iteration")
    ax.legend()

    # Win rate
    ax = axes[0, 1]
    if evaluation:
        ax.plot([r["iteration"] for r in evaluation],
                [r["win_rate"] for r in evaluation],
                "o-", color="tab:green", markersize=5)
    ax.axhline(0.25, color="gray", linestyle="--", alpha=0.7)
    ax.set_title("Win Rate vs Random")
    ax.set_xlabel("Iteration")
    ax.set_ylim(-0.05, 1.05)

    # Scores
    ax = axes[1, 0]
    if self_play:
        ax.fill_between(sp_iters, avg_worst, avg_best,
                        alpha=0.2, color="tab:blue")
        ax.plot(sp_iters, avg_mean, "o-", color="tab:blue", markersize=3)
    ax.set_title("Self-Play Scores")
    ax.set_xlabel("Iteration")

    # Game length
    ax = axes[1, 1]
    if self_play:
        ax.plot(sp_iters, avg_moves, "o-", color="tab:orange", markersize=3)
    ax.set_title("Game Length")
    ax.set_xlabel("Iteration")

    fig.suptitle("BlokusAI Training Dashboard", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path / "training_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path / 'training_summary.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--out-dir", default="plots")
    args = parser.parse_args()
    plot_all(args.log_dir, args.out_dir)
