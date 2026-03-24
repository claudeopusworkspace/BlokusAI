"""Demo: play a full Blokus game with random move selection."""

from __future__ import annotations

import random
import time

from .game import Game
from .board import NUM_PLAYERS
from .display import print_board


def play_random_game(seed: int | None = None, verbose: bool = True) -> Game:
    """Play a complete game where each player picks a random legal move."""
    rng = random.Random(seed)
    game = Game()
    move_count = 0

    if verbose:
        print("=== Blokus — Random Play Demo ===\n")

    while not game.game_over:
        moves = game.get_legal_moves()
        player = game.current_player
        if moves:
            move = rng.choice(moves)
            if verbose:
                print(
                    f"  P{player} plays {move.piece_name} "
                    f"(ori {move.orientation_idx}) at ({move.row},{move.col})"
                )
            game.apply_move(move)
            move_count += 1
        else:
            if verbose:
                print(f"  P{player} passes (no legal moves)")
            game.skip_turn()

    if verbose:
        print(f"\n--- Game over after {move_count} placements ---\n")
        print_board(game.board)
        print()
        scores = game.get_scores()
        for p in range(NUM_PLAYERS):
            remaining = len(game.board.remaining_pieces[p])
            print(
                f"  Player {p}: score {scores[p]:>4}  "
                f"({remaining} pieces remaining)"
            )
        winner = game.get_winner()
        print(f"\n  Winner: Player {winner} (score {scores[winner]})")

    return game


if __name__ == "__main__":
    t0 = time.perf_counter()
    play_random_game(seed=42)
    elapsed = time.perf_counter() - t0
    print(f"\n  Elapsed: {elapsed:.2f}s")
