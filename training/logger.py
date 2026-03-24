"""Structured JSON-Lines logger for training metrics.

Produces four log streams (one file each):

* **training.jsonl** — per-batch loss and learning-rate.
* **self_play.jsonl** — per-game self-play results.
* **evaluation.jsonl** — periodic win-rate evaluations.
* **iterations.jsonl** — per-iteration summary.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class TrainingLogger:
    """Append-only JSONL logger that writes to ``log_dir``."""

    def __init__(self, log_dir: str = "logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._files: dict[str, Any] = {}
        for name in ("training", "self_play", "evaluation", "iterations"):
            path = self.log_dir / f"{name}.jsonl"
            self._files[name] = open(path, "a", buffering=1)  # line-buffered

    # ------------------------------------------------------------------ #
    # Core writer
    # ------------------------------------------------------------------ #

    def _write(self, stream: str, record: dict) -> None:
        record.setdefault("timestamp", time.time())
        self._files[stream].write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------ #
    # Convenience methods
    # ------------------------------------------------------------------ #

    def log_training_batch(
        self,
        iteration: int,
        batch: int,
        policy_loss: float,
        value_loss: float,
        total_loss: float,
        lr: float,
    ) -> None:
        self._write("training", {
            "iteration": iteration,
            "batch": batch,
            "policy_loss": round(policy_loss, 6),
            "value_loss": round(value_loss, 6),
            "total_loss": round(total_loss, 6),
            "lr": lr,
        })

    def log_self_play_game(
        self,
        iteration: int,
        game_id: int,
        scores: list[int],
        winner: int,
        num_moves: int,
        duration_s: float,
    ) -> None:
        self._write("self_play", {
            "iteration": iteration,
            "game_id": game_id,
            "scores": scores,
            "winner": winner,
            "num_moves": num_moves,
            "duration_s": round(duration_s, 3),
        })

    def log_evaluation(
        self,
        iteration: int,
        opponent: str,
        games: int,
        wins: int,
        avg_score: float,
    ) -> None:
        self._write("evaluation", {
            "iteration": iteration,
            "opponent": opponent,
            "games": games,
            "wins": wins,
            "win_rate": round(wins / games, 4) if games else 0.0,
            "avg_score": round(avg_score, 2),
        })

    def log_iteration(
        self,
        iteration: int,
        avg_policy_loss: float,
        avg_value_loss: float,
        buffer_size: int,
        self_play_time_s: float,
        training_time_s: float,
        eval_win_rate: float | None = None,
    ) -> None:
        self._write("iterations", {
            "iteration": iteration,
            "avg_policy_loss": round(avg_policy_loss, 6),
            "avg_value_loss": round(avg_value_loss, 6),
            "buffer_size": buffer_size,
            "self_play_time_s": round(self_play_time_s, 2),
            "training_time_s": round(training_time_s, 2),
            "eval_win_rate": round(eval_win_rate, 4) if eval_win_rate is not None else None,
        })

    def close(self) -> None:
        for f in self._files.values():
            f.close()
