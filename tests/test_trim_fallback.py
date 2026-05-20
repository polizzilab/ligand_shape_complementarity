import numpy as np

from shape_complementarity import (
    Surface,
    extract_buried_surface,
    score_surfaces,
    trim_interface_surface,
    trim_interface_surface_guarded,
)
from shape_complementarity.score import _target_surface_for_direction


def _normalize_rows(points: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1.0e-8)
    return points / norms


def _make_surface(points, buried_flags):
    points_arr = np.asarray(points, dtype=np.float32)
    normals = _normalize_rows(points_arr + 0.1)
    areas = np.ones(points_arr.shape[0], dtype=np.float32)
    probe_centers = points_arr + normals
    atom_indices = np.zeros(points_arr.shape[0], dtype=np.int64)
    buried = np.asarray(buried_flags, dtype=bool)
    return Surface(points_arr, normals, areas, probe_centers, atom_indices, buried)


def test_strict_trim_can_be_smaller_than_buried():
    accessible = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]]
    buried_rim = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]]
    buried_core = [[2.0, 2.0, 0.0], [2.0, 2.5, 0.0], [2.0, 3.0, 0.0], [2.0, 3.5, 0.0]]
    points = accessible + buried_rim + buried_core
    buried_flags = [False] * len(accessible) + [True] * (len(buried_rim) + len(buried_core))
    marked = _make_surface(points, buried_flags)

    trimmed = trim_interface_surface(marked, band=1.5)
    buried = extract_buried_surface(marked)

    assert trimmed.points.shape[0] < buried.points.shape[0]


def test_target_fallback_uses_buried_when_trimmed_sparse():
    protein_trimmed = _make_surface([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], [True, True])
    protein_marked = _make_surface(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0]],
        [True, True, True, True, True],
    )
    ligand = _make_surface([[0.0, 0.0, 3.0], [0.0, 0.0, 3.5], [0.0, 0.0, 4.0]], [True, True, True])

    target = _target_surface_for_direction(
        protein_trimmed,
        protein_marked,
        min_target_points=4,
        near_source=ligand,
        near_source_dist=6.0,
    )
    assert target.points.shape[0] > protein_trimmed.points.shape[0]


def test_trim_guard_reverts_to_buried_when_trim_too_sparse():
    accessible = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]]
    buried_rim = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    buried_core = [[2.0, 2.0, 0.0], [2.0, 2.5, 0.0], [2.0, 3.0, 0.0], [2.0, 3.5, 0.0], [2.0, 4.0, 0.0]]
    points = accessible + buried_rim + buried_core
    buried_flags = [False] * len(accessible) + [True] * (len(buried_rim) + len(buried_core))
    marked = _make_surface(points, buried_flags)

    guarded = trim_interface_surface_guarded(marked, band=1.5, min_trimmed_points=6)
    buried = extract_buried_surface(marked)

    assert guarded.points.shape[0] == buried.points.shape[0]


def test_reverse_uses_interface_matched_source_when_ligand_larger():
    protein = _make_surface([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]], [True, True, True])
    ligand = _make_surface(
        [[0.0, 0.0, 3.0], [0.0, 0.0, 3.5], [0.0, 0.0, 4.0], [0.0, 0.0, 4.5], [0.0, 0.0, 5.0]],
        [True, True, True, True, True],
    )

    result = score_surfaces(protein, ligand, min_target_points=2, fallback_nn_dist=6.0)
    assert result.reverse.n_points < ligand.points.shape[0]
    assert result.reverse.n_points > 0

    protein_trimmed = _make_surface([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], [True, True])
    protein_marked = _make_surface(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0]],
        [True, True, True, True, True],
    )
    ligand = _make_surface(
        [[0.0, 0.0, 3.0], [0.0, 0.0, 3.5], [0.0, 0.0, 4.0], [5.0, 5.0, 5.0]],
        [True, True, True, True],
    )

    result = score_surfaces(
        protein_trimmed,
        ligand,
        marked_a=protein_marked,
        min_target_points=4,
        fallback_nn_dist=6.0,
    )
    assert result.forward.n_points > 0
    assert result.reverse.n_points > 0
    assert result.reverse.n_points < ligand.points.shape[0]
    assert result.sc >= result.forward.sc * 0.5
