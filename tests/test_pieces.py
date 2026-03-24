"""Tests for piece definitions and orientation generation."""

import pytest

from blokus.pieces import (
    PIECES,
    PIECE_NAMES,
    PIECE_INDEX,
    PieceData,
    Orientation,
    _parse_grid,
    _normalize,
    _generate_orientations,
)


class TestPieceCatalog:
    def test_piece_count(self):
        assert len(PIECES) == 21

    def test_piece_names_order(self):
        assert len(PIECE_NAMES) == 21
        assert PIECE_NAMES[0] == "I1"
        assert PIECE_NAMES[-1] == "Z5"

    def test_piece_index_bijects(self):
        for i, name in enumerate(PIECE_NAMES):
            assert PIECE_INDEX[name] == i

    def test_total_squares(self):
        total = sum(PIECES[n].size for n in PIECE_NAMES)
        assert total == 89  # 1+2+3+3+4*5+5*12

    def test_piece_sizes(self):
        sizes = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for p in PIECES.values():
            sizes[p.size] += 1
        assert sizes == {1: 1, 2: 1, 3: 2, 4: 5, 5: 12}


class TestOrientations:
    EXPECTED = {
        "I1": 1, "I2": 2, "I3": 2, "V3": 4,
        "I4": 2, "L4": 8, "T4": 4, "S4": 4, "O4": 1,
        "F5": 8, "I5": 2, "L5": 8, "N5": 8, "P5": 8,
        "T5": 4, "U5": 4, "V5": 4, "W5": 4, "X5": 1, "Y5": 8, "Z5": 4,
    }

    def test_orientation_counts(self):
        for name, expected in self.EXPECTED.items():
            piece = PIECES[name]
            assert len(piece.orientations) == expected, (
                f"{name}: expected {expected}, got {len(piece.orientations)}"
            )

    def test_total_orientations(self):
        total = sum(len(PIECES[n].orientations) for n in PIECE_NAMES)
        assert total == 91

    def test_all_orientations_unique(self):
        """No two orientations of the same piece should be identical."""
        for name in PIECE_NAMES:
            piece = PIECES[name]
            seen = set()
            for ori in piece.orientations:
                key = ori.cells  # already a sorted tuple
                assert key not in seen, f"{name} has duplicate orientation {key}"
                seen.add(key)

    def test_orientation_cell_counts(self):
        """Each orientation should have the same number of cells as the piece."""
        for name in PIECE_NAMES:
            piece = PIECES[name]
            for ori in piece.orientations:
                assert len(ori.cells) == piece.size

    def test_orientation_is_normalized(self):
        """All orientations should have min row = 0 and min col = 0."""
        for name in PIECE_NAMES:
            for ori in PIECES[name].orientations:
                rows = [r for r, _ in ori.cells]
                cols = [c for _, c in ori.cells]
                assert min(rows) == 0
                assert min(cols) == 0


class TestOrientationNeighbors:
    def test_monomino_neighbors(self):
        ori = PIECES["I1"].orientations[0]
        assert ori.cells == ((0, 0),)
        assert ori.edge_neighbors == frozenset({(-1, 0), (1, 0), (0, -1), (0, 1)})
        assert ori.corner_neighbors == frozenset(
            {(-1, -1), (-1, 1), (1, -1), (1, 1)}
        )

    def test_domino_neighbors(self):
        # I2 horizontal: (0,0),(0,1)
        ori = PIECES["I2"].orientations[0]
        assert len(ori.cells) == 2
        # Edge neighbors: 6 cells around the domino
        assert len(ori.edge_neighbors) == 6
        # Corner neighbors: 4 diagonal corners not touching edges
        assert len(ori.corner_neighbors) == 4

    def test_no_overlap_between_neighbors_and_cells(self):
        for name in PIECE_NAMES:
            for ori in PIECES[name].orientations:
                assert not ori.cell_set & ori.edge_neighbors
                assert not ori.cell_set & ori.corner_neighbors
                assert not ori.edge_neighbors & ori.corner_neighbors


class TestParseGrid:
    def test_simple(self):
        cells = _parse_grid("XX\nX.")
        assert cells == frozenset({(0, 0), (0, 1), (1, 0)})

    def test_single_cell(self):
        cells = _parse_grid("X")
        assert cells == frozenset({(0, 0)})
