"""Demo: play Blokus games with various player configurations."""

from __future__ import annotations

import argparse
import time

from .board import NUM_PLAYERS
from .players import MCTSPlayer, RandomPlayer, play_game


def main() -> None:
    parser = argparse.ArgumentParser(description="Blokus demo game")
    parser.add_argument(
        "--mcts", type=int, default=0, metavar="SIMS",
        help="Number of MCTS simulations for player 0 (0 = all random)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.mcts > 0:
        players = [
            MCTSPlayer(num_simulations=args.mcts, seed=args.seed),
            RandomPlayer(seed=args.seed + 1),
            RandomPlayer(seed=args.seed + 2),
            RandomPlayer(seed=args.seed + 3),
        ]
    else:
        players = [RandomPlayer(seed=args.seed + i) for i in range(NUM_PLAYERS)]

    t0 = time.perf_counter()
    play_game(players, verbose=not args.quiet)
    elapsed = time.perf_counter() - t0
    if not args.quiet:
        print(f"\n  Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
