"""Player interfaces for Blokus games.

Provides a common protocol and concrete implementations for random and
MCTS-driven players, plus a game runner that accepts any mix of players.
"""

from __future__ import annotations

import random
import time
from typing import List, Optional, Protocol

from .board import Move, NUM_PLAYERS
from .game import Game


# ------------------------------------------------------------------ #
# Player protocol
# ------------------------------------------------------------------ #

class Player(Protocol):
    """Anything that can choose a move given a game state."""

    name: str

    def choose_move(self, game: Game) -> Move: ...


# ------------------------------------------------------------------ #
# Concrete players
# ------------------------------------------------------------------ #

class RandomPlayer:
    """Uniform-random legal-move selection."""

    def __init__(self, seed: int | None = None) -> None:
        self.name = "Random"
        self.rng = random.Random(seed)

    def choose_move(self, game: Game) -> Move:
        return self.rng.choice(game.get_legal_moves())


class MCTSPlayer:
    """MCTS-driven player."""

    def __init__(
        self,
        num_simulations: int = 200,
        exploration: float = 1.41,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> None:
        from mcts import MCTS

        self.name = f"MCTS({num_simulations})"
        self.temperature = temperature
        self.mcts = MCTS(
            num_simulations=num_simulations,
            exploration=exploration,
            rollout_rng=random.Random(seed),
        )
        self._last_stats: dict = {}

    def choose_move(self, game: Game) -> Move:
        move, stats = self.mcts.search(game, temperature=self.temperature)
        self._last_stats = stats
        return move


# ------------------------------------------------------------------ #
# Game runner
# ------------------------------------------------------------------ #

def play_game(
    players: List[Player],
    verbose: bool = False,
) -> Game:
    """Run a complete Blokus game with the given players (one per seat).

    Returns the finished :class:`Game`.
    """
    assert len(players) == NUM_PLAYERS
    game = Game()
    move_count = 0

    if verbose:
        from .display import print_board

        names = ", ".join(
            f"P{i}={players[i].name}" for i in range(NUM_PLAYERS)
        )
        print(f"=== Blokus  [{names}] ===\n")

    while not game.game_over:
        p = game.current_player
        move = players[p].choose_move(game)

        if verbose:
            extra = ""
            if hasattr(players[p], "_last_stats") and players[p]._last_stats:
                s = players[p]._last_stats
                extra = (
                    f"  ({s['elapsed_s']:.2f}s, "
                    f"{s['sims_per_sec']:.0f} sims/s, "
                    f"val={s['best_value']:.3f})"
                )
            print(
                f"  P{p} ({players[p].name}) plays {move.piece_name} "
                f"ori={move.orientation_idx} at ({move.row},{move.col}){extra}"
            )

        game.apply_move(move)
        move_count += 1

    if verbose:
        print(f"\n--- Game over after {move_count} moves ---\n")
        print_board(game.board)
        print()
        scores = game.get_scores()
        for i in range(NUM_PLAYERS):
            rem = len(game.board.remaining_pieces[i])
            print(
                f"  P{i} ({players[i].name:>12}): "
                f"score {scores[i]:>4}  ({rem} pieces left)"
            )
        w = game.get_winner()
        print(f"\n  Winner: P{w} ({players[w].name}) — score {scores[w]}")

    return game
