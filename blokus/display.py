"""Terminal display for Blokus boards."""

from __future__ import annotations

from .board import BoardState, BOARD_SIZE

# ANSI colour codes
_PLAYER_FG = (
    "\033[94m",  # Player 0 — Blue
    "\033[93m",  # Player 1 — Yellow
    "\033[91m",  # Player 2 — Red
    "\033[92m",  # Player 3 — Green
)
_EMPTY = "\033[90m"  # dim grey
_RESET = "\033[0m"

_PLAYER_CHAR = ("B", "Y", "R", "G")


def render_board(board: BoardState, use_color: bool = True) -> str:
    """Return a multi-line string representation of the board."""
    lines: list[str] = []

    # Column header
    hdr = "    " + " ".join(f"{c:>2}" for c in range(BOARD_SIZE))
    lines.append(hdr)

    for r in range(BOARD_SIZE):
        parts: list[str] = [f"{r:>2}  "]
        for c in range(BOARD_SIZE):
            val = int(board.grid[r, c])
            if val == 0:
                if use_color:
                    parts.append(f"{_EMPTY} .{_RESET}")
                else:
                    parts.append(" .")
            else:
                p = val - 1
                ch = _PLAYER_CHAR[p]
                if use_color:
                    parts.append(f"{_PLAYER_FG[p]} {ch}{_RESET}")
                else:
                    parts.append(f" {ch}")
        lines.append(" ".join(parts) if not use_color else "".join(parts))

    return "\n".join(lines)


def print_board(board: BoardState, use_color: bool = True) -> None:
    """Print the board to stdout."""
    print(render_board(board, use_color=use_color))
