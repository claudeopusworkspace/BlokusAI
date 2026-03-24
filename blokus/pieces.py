"""Piece definitions and orientation generation for Blokus.

Each player has 21 polyomino pieces (1 monomino through 12 pentominoes).
Pieces can be rotated and flipped, generating up to 8 unique orientations each.
"""

from __future__ import annotations

from typing import Tuple, FrozenSet, List

# Type aliases
Cell = Tuple[int, int]
Cells = FrozenSet[Cell]

# Orthogonal directions (edge neighbors)
EDGE_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))
# Diagonal directions (corner neighbors)
CORNER_DELTAS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


class Orientation:
    """A specific orientation of a piece, with precomputed neighbor data.

    Cells are stored as (row, col) offsets from the placement anchor (0,0).
    Edge and corner neighbors are cells adjacent to the piece but not part of it.
    """

    __slots__ = (
        "cells",
        "cell_set",
        "edge_neighbors",
        "corner_neighbors",
        "height",
        "width",
    )

    def __init__(self, cells: Cells):
        self.cells: Tuple[Cell, ...] = tuple(sorted(cells))
        self.cell_set: FrozenSet[Cell] = frozenset(cells)

        edge_set: set[Cell] = set()
        corner_set: set[Cell] = set()

        for r, c in cells:
            for dr, dc in EDGE_DELTAS:
                nb = (r + dr, c + dc)
                if nb not in self.cell_set:
                    edge_set.add(nb)
            for dr, dc in CORNER_DELTAS:
                nb = (r + dr, c + dc)
                if nb not in self.cell_set:
                    corner_set.add(nb)

        self.edge_neighbors: FrozenSet[Cell] = frozenset(edge_set)
        self.corner_neighbors: FrozenSet[Cell] = frozenset(corner_set - edge_set)

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        self.height: int = max(rows) + 1
        self.width: int = max(cols) + 1

    def __repr__(self) -> str:
        return f"Orientation({self.cells})"


# ------------------------------------------------------------------ #
# Orientation generation via rotation and reflection
# ------------------------------------------------------------------ #

def _normalize(cells: Cells) -> Cells:
    """Translate cells so the minimum row and column are both 0."""
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return frozenset((r - min_r, c - min_c) for r, c in cells)


def _rotate_90(cells: Cells) -> Cells:
    """Rotate 90 degrees clockwise: (r, c) -> (c, -r)."""
    return frozenset((c, -r) for r, c in cells)


def _flip_h(cells: Cells) -> Cells:
    """Reflect horizontally: (r, c) -> (r, -c)."""
    return frozenset((r, -c) for r, c in cells)


def _generate_orientations(cells: Cells) -> List[Orientation]:
    """Generate all unique orientations of a piece (up to 8)."""
    seen: set[Tuple[Cell, ...]] = set()
    orientations: List[Orientation] = []
    current = cells
    for _ in range(4):
        for variant in (current, _flip_h(current)):
            normalized = _normalize(variant)
            key = tuple(sorted(normalized))
            if key not in seen:
                seen.add(key)
                orientations.append(Orientation(normalized))
        current = _rotate_90(current)
    return orientations


# ------------------------------------------------------------------ #
# Piece grid parser
# ------------------------------------------------------------------ #

def _parse_grid(grid_str: str) -> Cells:
    """Parse a text grid into cell offsets. 'X' = filled, '.' = empty."""
    cells: set[Cell] = set()
    for r, line in enumerate(grid_str.strip().split("\n")):
        for c, ch in enumerate(line.strip()):
            if ch == "X":
                cells.add((r, c))
    return frozenset(cells)


# ------------------------------------------------------------------ #
# The 21 Blokus pieces
# ------------------------------------------------------------------ #

PIECE_DEFS: dict[str, str] = {
    # --- 1 cell ---
    "I1": "X",
    # --- 2 cells ---
    "I2": "XX",
    # --- 3 cells ---
    "I3": "XXX",
    "V3": "XX\n"
          "X.",
    # --- 4 cells ---
    "I4": "XXXX",
    "L4": "XXX\n"
          "X..",
    "T4": "XXX\n"
          ".X.",
    "S4": ".XX\n"
          "XX.",
    "O4": "XX\n"
          "XX",
    # --- 5 cells ---
    "F5": ".XX\n"
          "XX.\n"
          ".X.",
    "I5": "XXXXX",
    "L5": "X.\n"
          "X.\n"
          "X.\n"
          "XX",
    "N5": ".X\n"
          ".X\n"
          "XX\n"
          "X.",
    "P5": "XX\n"
          "XX\n"
          "X.",
    "T5": "XXX\n"
          ".X.\n"
          ".X.",
    "U5": "X.X\n"
          "XXX",
    "V5": "X..\n"
          "X..\n"
          "XXX",
    "W5": "X..\n"
          "XX.\n"
          ".XX",
    "X5": ".X.\n"
          "XXX\n"
          ".X.",
    "Y5": ".X\n"
          "XX\n"
          ".X\n"
          ".X",
    "Z5": "XX.\n"
          ".X.\n"
          ".XX",
}


class PieceData:
    """Metadata for a single Blokus piece, including all orientations."""

    __slots__ = ("name", "cells", "size", "orientations")

    def __init__(self, name: str, cells: Cells):
        self.name: str = name
        self.cells: Cells = cells
        self.size: int = len(cells)
        self.orientations: List[Orientation] = _generate_orientations(cells)

    def __repr__(self) -> str:
        return (
            f"PieceData({self.name!r}, size={self.size}, "
            f"orientations={len(self.orientations)})"
        )


# ------------------------------------------------------------------ #
# Build the global piece catalog
# ------------------------------------------------------------------ #

PIECES: dict[str, PieceData] = {}
PIECE_NAMES: List[str] = []

for _name, _grid in PIECE_DEFS.items():
    _cells = _parse_grid(_grid)
    PIECES[_name] = PieceData(_name, _cells)
    PIECE_NAMES.append(_name)

# Sanity checks
assert len(PIECES) == 21, f"Expected 21 pieces, got {len(PIECES)}"
assert sum(p.size for p in PIECES.values()) == 89, "Total squares should be 89"

# Expected orientation counts per piece (rotation + reflection, deduplicated)
_EXPECTED_ORIENTATIONS = {
    "I1": 1, "I2": 2, "I3": 2, "V3": 4,
    "I4": 2, "L4": 8, "T4": 4, "S4": 4, "O4": 1,
    "F5": 8, "I5": 2, "L5": 8, "N5": 8, "P5": 8,
    "T5": 4, "U5": 4, "V5": 4, "W5": 4, "X5": 1, "Y5": 8, "Z5": 4,
}
for _name, _expected in _EXPECTED_ORIENTATIONS.items():
    _actual = len(PIECES[_name].orientations)
    assert _actual == _expected, (
        f"{_name}: expected {_expected} orientations, got {_actual}"
    )

# Piece name -> integer index (for neural network encoding)
PIECE_INDEX: dict[str, int] = {name: i for i, name in enumerate(PIECE_NAMES)}
NUM_PIECES = len(PIECE_NAMES)
