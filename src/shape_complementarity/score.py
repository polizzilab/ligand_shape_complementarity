"""Rosetta-inspired shape-complementarity scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import nearest_neighbors
from .surface import Surface, build_contact_surface, extract_buried_surface, mark_buried_by_atoms, trim_interface_surface


@dataclass(frozen=True)
class DirectionalScore:
    sc: float
    distance: float
    area: float
    n_points: int
    mean_sc: float
    mean_distance: float


@dataclass(frozen=True)
class ShapeComplementarityResult:
    sc: float
    distance: float
    area: float
    forward: DirectionalScore
    reverse: DirectionalScore
    n_surface_points_a: int
    n_surface_points_b: int
    n_trimmed_points_a: int
    n_trimmed_points_b: int


@dataclass(frozen=True)
class InterfaceSurfaces:
    """Intermediate surfaces used to compute shape complementarity."""

    surface_a: Surface
    surface_b: Surface
    marked_a: Surface
    marked_b: Surface
    trimmed_a: Surface
    trimmed_b: Surface
    marked_a_full: Surface | None = None


def _finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _median(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, 0.5))


def _nearest_directional_score(
    source: Surface,
    target: Surface,
    *,
    weight: float = 0.5,
    max_nn_dist: float = 6.0,
    chunk_size: int = 65536,
) -> DirectionalScore:
    """Score each source dot against its nearest target dot."""
    if source.points.shape[0] == 0 or target.points.shape[0] == 0:
        return DirectionalScore(
            sc=float("nan"),
            distance=float("nan"),
            area=float(source.areas.sum()),
            n_points=int(source.points.shape[0]),
            mean_sc=float("nan"),
            mean_distance=float("nan"),
        )

    distances, indices = nearest_neighbors(source.points, target.points, chunk_size=chunk_size)

    in_range = distances <= max_nn_dist
    distances = distances[in_range]
    indices = indices[in_range]

    if distances.shape[0] == 0:
        return DirectionalScore(
            sc=float("nan"),
            distance=float("nan"),
            area=float(source.areas.sum()),
            n_points=int(source.points.shape[0]),
            mean_sc=float("nan"),
            mean_distance=float("nan"),
        )

    normal_dot = np.sum(source.normals[in_range] * target.normals[indices], axis=1)
    normal_dot = np.clip(normal_dot, -1.0, 1.0)
    sc_values = -normal_dot * np.exp(-(distances * distances) * weight)

    return DirectionalScore(
        sc=_median(sc_values),
        distance=_median(distances),
        area=float(source.areas.sum()),
        n_points=int(distances.shape[0]),
        mean_sc=float(sc_values.mean()),
        mean_distance=float(distances.mean()),
    )


def _target_surface_for_direction(
    trimmed: Surface,
    marked: Surface | None,
    *,
    marked_fallback: Surface | None = None,
    min_target_points: int,
    near_source: Surface | None = None,
    near_source_dist: float = 6.0,
) -> Surface:
    """Prefer trimmed dots; fall back to buried, then full-molecule buried."""
    if trimmed.points.shape[0] >= min_target_points:
        return trimmed

    best = trimmed
    for candidate in (marked, marked_fallback):
        if candidate is None:
            continue
        buried = extract_buried_surface(candidate)
        if buried.points.shape[0] > best.points.shape[0]:
            best = buried
        if best.points.shape[0] >= min_target_points:
            break

    if (
        near_source is not None
        and best.points.shape[0] > trimmed.points.shape[0]
        and near_source.points.shape[0] > 0
        and best.points.shape[0] > 0
    ):
        from .geometry import min_distance_to_points

        keep = min_distance_to_points(best.points, near_source.points) <= near_source_dist
        if keep.any():
            best = Surface(
                best.points[keep],
                best.normals[keep],
                best.areas[keep],
                best.probe_centers[keep],
                best.atom_indices[keep],
                best.buried[keep] if best.buried is not None else None,
            )
    return best


def _interface_matched_source(source: Surface, target: Surface) -> Surface:
    """Keep source dots that are the nearest neighbour of at least one target dot."""
    if source.points.shape[0] == 0 or target.points.shape[0] == 0:
        return source.empty_like()
    _, nn_idx = nearest_neighbors(target.points, source.points)
    keep = np.zeros(source.points.shape[0], dtype=bool)
    keep[np.unique(nn_idx)] = True
    if not keep.any():
        return source.empty_like()
    return Surface(
        source.points[keep],
        source.normals[keep],
        source.areas[keep],
        source.probe_centers[keep],
        source.atom_indices[keep],
        source.buried[keep] if source.buried is not None else None,
    )


@dataclass(frozen=True)
class ScoringSurfaces:
    """Exact source/target surfaces used in each scoring direction."""

    forward_source: Surface
    forward_target: Surface
    reverse_source: Surface
    reverse_target: Surface
    forward_nn_cap: float
    reverse_nn_cap: float


def resolve_scoring_surfaces(
    surface_a: Surface,
    surface_b: Surface,
    *,
    marked_a: Surface | None = None,
    marked_b: Surface | None = None,
    marked_a_fallback: Surface | None = None,
    marked_b_fallback: Surface | None = None,
    min_target_points: int = 2000,
    fallback_nn_dist: float = 6.0,
    max_nn_dist: float | None = None,
) -> ScoringSurfaces:
    """Return the source/target surfaces passed to directional scoring."""
    target_b = _target_surface_for_direction(
        surface_b,
        marked_b,
        marked_fallback=marked_b_fallback,
        min_target_points=min_target_points,
        near_source=surface_a,
        near_source_dist=fallback_nn_dist,
    )
    target_a = _target_surface_for_direction(
        surface_a,
        marked_a,
        marked_fallback=marked_a_fallback,
        min_target_points=min_target_points,
        near_source=surface_b,
        near_source_dist=fallback_nn_dist,
    )
    reverse_nn_cap = fallback_nn_dist if target_a.points.shape[0] > surface_a.points.shape[0] else float("inf")
    if max_nn_dist is not None:
        reverse_nn_cap = min(reverse_nn_cap, max_nn_dist)
    forward_nn_cap = float("inf") if max_nn_dist is None else max_nn_dist
    use_matched_reverse = target_a.points.shape[0] > 0 and (
        surface_a.points.shape[0] < min_target_points or surface_b.points.shape[0] > surface_a.points.shape[0]
    )
    reverse_source = _interface_matched_source(surface_b, target_a) if use_matched_reverse else surface_b
    return ScoringSurfaces(
        forward_source=surface_a,
        forward_target=target_b,
        reverse_source=reverse_source,
        reverse_target=target_a,
        forward_nn_cap=forward_nn_cap,
        reverse_nn_cap=reverse_nn_cap,
    )


def score_surfaces(
    surface_a: Surface,
    surface_b: Surface,
    *,
    marked_a: Surface | None = None,
    marked_b: Surface | None = None,
    marked_a_fallback: Surface | None = None,
    marked_b_fallback: Surface | None = None,
    weight: float = 0.5,
    max_nn_dist: float | None = None,
    min_target_points: int = 2000,
    fallback_nn_dist: float = 6.0,
    chunk_size: int = 65536,
) -> ShapeComplementarityResult:
    """Score two pre-trimmed interface surfaces."""
    resolved = resolve_scoring_surfaces(
        surface_a,
        surface_b,
        marked_a=marked_a,
        marked_b=marked_b,
        marked_a_fallback=marked_a_fallback,
        marked_b_fallback=marked_b_fallback,
        min_target_points=min_target_points,
        fallback_nn_dist=fallback_nn_dist,
        max_nn_dist=max_nn_dist,
    )
    forward = _nearest_directional_score(
        resolved.forward_source,
        resolved.forward_target,
        weight=weight,
        max_nn_dist=resolved.forward_nn_cap,
        chunk_size=chunk_size,
    )
    reverse = _nearest_directional_score(
        resolved.reverse_source,
        resolved.reverse_target,
        weight=weight,
        max_nn_dist=resolved.reverse_nn_cap,
        chunk_size=chunk_size,
    )

    if forward.n_points == 0 or reverse.n_points == 0:
        sc = float("nan")
        distance = float("nan")
    else:
        fwd_sc = forward.sc if _finite(forward.sc) else 0.0
        rev_sc = reverse.sc if _finite(reverse.sc) else 0.0
        sc = max(0.0, (fwd_sc + rev_sc) / 2.0)
        distance = (forward.distance + reverse.distance) / 2.0

    return ShapeComplementarityResult(
        sc=sc,
        distance=distance,
        area=forward.area + reverse.area,
        forward=forward,
        reverse=reverse,
        n_surface_points_a=int(surface_a.points.shape[0]),
        n_surface_points_b=int(surface_b.points.shape[0]),
        n_trimmed_points_a=int(surface_a.points.shape[0]),
        n_trimmed_points_b=int(surface_b.points.shape[0]),
    )


def score_interface_surfaces(
    surfaces: InterfaceSurfaces,
    *,
    weight: float = 0.5,
    max_nn_dist: float | None = None,
    min_target_points: int = 2000,
    fallback_nn_dist: float = 6.0,
    chunk_size: int = 65536,
) -> ShapeComplementarityResult:
    """Score pre-built interface surfaces, including buried-dot fallbacks."""
    result = score_surfaces(
        surfaces.trimmed_a,
        surfaces.trimmed_b,
        marked_a=surfaces.marked_a,
        marked_b=surfaces.marked_b,
        marked_a_fallback=surfaces.marked_a_full,
        marked_b_fallback=None,
        weight=weight,
        max_nn_dist=max_nn_dist,
        min_target_points=min_target_points,
        fallback_nn_dist=fallback_nn_dist,
        chunk_size=chunk_size,
    )
    return ShapeComplementarityResult(
        sc=result.sc,
        distance=result.distance,
        area=result.area,
        forward=result.forward,
        reverse=result.reverse,
        n_surface_points_a=int(surfaces.surface_a.points.shape[0]),
        n_surface_points_b=int(surfaces.surface_b.points.shape[0]),
        n_trimmed_points_a=int(surfaces.trimmed_a.points.shape[0]),
        n_trimmed_points_b=int(surfaces.trimmed_b.points.shape[0]),
    )


def score_complex(
    atom_xyz_a,
    atom_radii_a,
    atom_xyz_b,
    atom_radii_b,
    *,
    n_sphere_points: int = 96,
    probe_radius: float = 1.7,
    band: float = 1.5,
    weight: float = 0.5,
    min_target_points: int = 2000,
    fallback_nn_dist: float = 6.0,
    chunk_size: int = 65536,
) -> ShapeComplementarityResult:
    """Build approximate surfaces for two molecules and compute SC."""
    surfaces = build_interface_surfaces(
        atom_xyz_a,
        atom_radii_a,
        atom_xyz_b,
        atom_radii_b,
        n_sphere_points=n_sphere_points,
        probe_radius=probe_radius,
        band=band,
        chunk_size=chunk_size,
    )
    return score_interface_surfaces(
        surfaces,
        weight=weight,
        min_target_points=min_target_points,
        fallback_nn_dist=fallback_nn_dist,
        chunk_size=chunk_size,
    )


def build_interface_surfaces(
    atom_xyz_a,
    atom_radii_a,
    atom_xyz_b,
    atom_radii_b,
    *,
    n_sphere_points: int = 96,
    probe_radius: float = 1.7,
    band: float = 1.5,
    chunk_size: int = 65536,
) -> InterfaceSurfaces:
    """Return all intermediate surfaces for scoring and visual debugging."""
    surface_a = build_contact_surface(
        atom_xyz_a,
        atom_radii_a,
        n_sphere_points=n_sphere_points,
        probe_radius=probe_radius,
        chunk_size=chunk_size,
    )
    surface_b = build_contact_surface(
        atom_xyz_b,
        atom_radii_b,
        n_sphere_points=n_sphere_points,
        probe_radius=probe_radius,
        chunk_size=chunk_size,
    )
    marked_a = mark_buried_by_atoms(
        surface_a,
        atom_xyz_b,
        atom_radii_b,
        probe_radius=probe_radius,
        chunk_size=chunk_size,
    )
    marked_b = mark_buried_by_atoms(
        surface_b,
        atom_xyz_a,
        atom_radii_a,
        probe_radius=probe_radius,
        chunk_size=chunk_size,
    )
    trimmed_a = trim_interface_surface(marked_a, band=band, chunk_size=chunk_size)
    trimmed_b = trim_interface_surface(marked_b, band=band, chunk_size=chunk_size)
    return InterfaceSurfaces(
        surface_a=surface_a,
        surface_b=surface_b,
        marked_a=marked_a,
        marked_b=marked_b,
        trimmed_a=trimmed_a,
        trimmed_b=trimmed_b,
    )
