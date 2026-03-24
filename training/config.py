"""Training hyper-parameters — single source of truth."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    # ---- network ----
    num_filters: int = 64
    num_res_blocks: int = 5
    policy_channels: int = 32
    value_hidden: int = 256

    # ---- MCTS ----
    num_simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temp_threshold: int = 10      # moves 1–N use τ=1; after that τ→0

    # ---- self-play ----
    games_per_iteration: int = 20

    # ---- training ----
    iterations: int = 50
    epochs_per_iteration: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    replay_buffer_size: int = 50_000

    # ---- evaluation ----
    eval_games: int = 20
    eval_interval: int = 5        # evaluate every N iterations
    eval_mcts_sims: int = 50      # sims for NN player during eval

    # ---- paths ----
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
