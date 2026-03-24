"""Benchmark: measure MCTS win rate against random players."""

from __future__ import annotations

import time
from blokus.board import NUM_PLAYERS
from blokus.game import Game
from blokus.players import MCTSPlayer, RandomPlayer, play_game


def run_benchmark(
    num_games: int = 20,
    mcts_sims: int = 50,
    mcts_seat: int = 0,
) -> None:
    wins = [0] * NUM_PLAYERS
    total_scores = [0] * NUM_PLAYERS
    t0 = time.perf_counter()

    for g in range(num_games):
        players = [RandomPlayer(seed=g * 10 + i) for i in range(NUM_PLAYERS)]
        players[mcts_seat] = MCTSPlayer(num_simulations=mcts_sims, seed=g * 100)

        game = play_game(players, verbose=False)
        scores = game.get_scores()
        winner = game.get_winner()
        wins[winner] += 1
        for p in range(NUM_PLAYERS):
            total_scores[p] += scores[p]

        elapsed = time.perf_counter() - t0
        avg_time = elapsed / (g + 1)
        print(
            f"  Game {g+1:>2}/{num_games}: "
            f"winner=P{winner} scores={scores}  "
            f"({avg_time:.1f}s/game)"
        )

    elapsed = time.perf_counter() - t0
    print(f"\n=== Results ({num_games} games, MCTS({mcts_sims}) at seat {mcts_seat}) ===")
    for p in range(NUM_PLAYERS):
        label = f"MCTS({mcts_sims})" if p == mcts_seat else "Random"
        avg = total_scores[p] / num_games
        print(
            f"  P{p} ({label:>10}): "
            f"{wins[p]:>2} wins ({100*wins[p]/num_games:.0f}%)  "
            f"avg score: {avg:.1f}"
        )
    print(f"\n  Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    run_benchmark(num_games=20, mcts_sims=50)
