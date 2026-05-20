"""Vectorized geometry helpers for surface construction and scoring."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


def as_float32(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def fibonacci_sphere(n_points: int, *, dtype: np.dtype = np.float32) -> np.ndarray:
    """Return approximately uniform unit vectors on a sphere."""
    if n_points <= 0:
        raise ValueError("n_points must be positive")

    i = np.arange(n_points, dtype=np.float64)
    z = 1.0 - (2.0 * i + 1.0) / float(n_points)
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = i * golden_angle
    dirs = np.stack((r * np.cos(theta), r * np.sin(theta), z), axis=-1)
    return dirs.astype(dtype, copy=False)


def min_distance_to_points(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Minimum distance from each source point to the target point set."""
    if source.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    if target.shape[0] == 0:
        return np.full(source.shape[0], np.inf, dtype=np.float32)
    tree = cKDTree(target)
    dists, _ = tree.query(source, k=1, workers=1)
    return np.asarray(dists, dtype=np.float32)


def nearest_neighbors(
    source: np.ndarray,
    target: np.ndarray,
    *,
    chunk_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest target index and distance for each source point."""
    if source.shape[0] == 0 or target.shape[0] == 0:
        return (
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    tree = cKDTree(target)
    distances = np.empty(source.shape[0], dtype=np.float32)
    indices = np.empty(source.shape[0], dtype=np.int64)
    for start in range(0, source.shape[0], chunk_size):
        stop = min(start + chunk_size, source.shape[0])
        dists, idx = tree.query(source[start:stop], k=1, workers=1)
        distances[start:stop] = dists
        indices[start:stop] = idx
    return distances, indices


def any_point_inside_spheres(
    query_points: np.ndarray,
    sphere_centers: np.ndarray,
    sphere_radii_sq: np.ndarray,
    *,
    ignore_sphere_for_query: np.ndarray | None = None,
    chunk_size: int = 65536,
) -> np.ndarray:
    """True when a query point lies inside at least one sphere."""
    hits = np.zeros(query_points.shape[0], dtype=bool)
    if query_points.shape[0] == 0 or sphere_centers.shape[0] == 0:
        return hits

    radii_sq = np.asarray(sphere_radii_sq, dtype=np.float32)
    for start in range(0, query_points.shape[0], chunk_size):
        stop = min(start + chunk_size, query_points.shape[0])
        chunk = query_points[start:stop]
        d2 = cdist(chunk, sphere_centers, metric="sqeuclidean")
        if ignore_sphere_for_query is not None:
            rows = np.arange(stop - start)
            cols = ignore_sphere_for_query[start:stop]
            d2[rows, cols] = np.inf
        hits[start:stop] = (d2 <= radii_sq).any(axis=1)
    return hits
