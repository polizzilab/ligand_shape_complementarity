"""Surface point generation and interface trimming."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import (
    any_point_inside_spheres,
    as_float32,
    fibonacci_sphere,
    min_distance_to_points,
)


@dataclass(frozen=True)
class Surface:
    """A point-cloud molecular surface."""

    points: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    probe_centers: np.ndarray
    atom_indices: np.ndarray
    buried: np.ndarray | None = None

    def empty_like(self) -> "Surface":
        return Surface(
            self.points[:0],
            self.normals[:0],
            self.areas[:0],
            self.probe_centers[:0],
            self.atom_indices[:0],
            None if self.buried is None else self.buried[:0],
        )


def filter_atoms_by_interface_sep(
    atom_xyz,
    atom_radii,
    other_atom_xyz,
    *,
    sep: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep atoms within ``sep`` Å of the opposite molecule (Rosetta SC default)."""
    xyz = as_float32(atom_xyz)
    radii = as_float32(atom_radii)
    other = as_float32(other_atom_xyz)
    if xyz.shape[0] == 0:
        return xyz, radii
    if other.shape[0] == 0:
        return xyz[:0], radii[:0]

    keep = min_distance_to_points(xyz, other) < sep
    return xyz[keep], radii[keep]


def interface_atom_mask(atom_xyz, other_atom_xyz, *, sep: float = 8.0) -> np.ndarray:
    """Boolean mask of atoms within ``sep`` Å of the opposite molecule."""
    xyz = as_float32(atom_xyz)
    other = as_float32(other_atom_xyz)
    if xyz.shape[0] == 0 or other.shape[0] == 0:
        return np.zeros(xyz.shape[0], dtype=bool)
    return min_distance_to_points(xyz, other) < sep


def filter_surface_by_atoms(surface: Surface, keep_atoms: np.ndarray) -> Surface:
    """Keep only surface dots whose parent atom is in ``keep_atoms``."""
    keep_atoms = np.asarray(keep_atoms, dtype=bool)
    if surface.points.shape[0] == 0:
        return surface.empty_like()
    keep = keep_atoms[surface.atom_indices]
    if not keep.any():
        return surface.empty_like()
    buried = surface.buried[keep] if surface.buried is not None else None
    return Surface(
        surface.points[keep],
        surface.normals[keep],
        surface.areas[keep],
        surface.probe_centers[keep],
        surface.atom_indices[keep],
        buried,
    )


def build_contact_surface(
    atom_xyz,
    atom_radii,
    *,
    n_sphere_points: int = 96,
    probe_radius: float = 1.7,
    chunk_size: int = 65536,
) -> Surface:
    """Build accessible atom-contact surface points."""
    xyz = as_float32(atom_xyz)
    radii = as_float32(atom_radii)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("atom_xyz must have shape [n_atoms, 3]")
    if radii.ndim != 1 or radii.shape[0] != xyz.shape[0]:
        raise ValueError("atom_radii must have shape [n_atoms]")

    if xyz.shape[0] == 0:
        empty3 = np.empty((0, 3), dtype=np.float32)
        empty1 = np.empty((0,), dtype=np.float32)
        empty_idx = np.empty((0,), dtype=np.int64)
        return Surface(empty3, empty3, empty1, empty3, empty_idx)

    dirs = fibonacci_sphere(n_sphere_points)
    points = xyz[:, None, :] + dirs[None, :, :] * radii[:, None, None]
    probe_centers = xyz[:, None, :] + dirs[None, :, :] * (radii[:, None, None] + probe_radius)

    flat_points = points.reshape(-1, 3)
    flat_probe_centers = probe_centers.reshape(-1, 3)
    flat_normals = np.tile(dirs, (xyz.shape[0], 1))
    atom_indices = np.repeat(np.arange(xyz.shape[0], dtype=np.int64), n_sphere_points)
    areas = np.repeat(4.0 * np.pi * radii * radii / float(n_sphere_points), n_sphere_points).astype(np.float32)

    occluded = any_point_inside_spheres(
        flat_probe_centers,
        xyz,
        (radii + probe_radius) ** 2,
        ignore_sphere_for_query=atom_indices,
        chunk_size=chunk_size,
    )
    keep = ~occluded
    return Surface(
        points=flat_points[keep],
        normals=flat_normals[keep],
        areas=areas[keep],
        probe_centers=flat_probe_centers[keep],
        atom_indices=atom_indices[keep],
    )


def mark_buried_by_atoms(
    surface: Surface,
    other_atom_xyz,
    other_atom_radii,
    *,
    probe_radius: float = 1.7,
    chunk_size: int = 65536,
) -> Surface:
    """Mark points whose solvent probe would overlap the opposite molecule."""
    other_xyz = as_float32(other_atom_xyz)
    other_radii = as_float32(other_atom_radii)
    buried = any_point_inside_spheres(
        surface.probe_centers,
        other_xyz,
        (other_radii + probe_radius) ** 2,
        chunk_size=chunk_size,
    )
    return Surface(
        surface.points,
        surface.normals,
        surface.areas,
        surface.probe_centers,
        surface.atom_indices,
        buried,
    )


def extract_buried_surface(surface: Surface) -> Surface:
    """Return the subset of surface dots marked buried."""
    if surface.buried is None:
        raise ValueError("surface must have buried flags")
    keep = surface.buried
    if not keep.any():
        return surface.empty_like()
    return Surface(
        surface.points[keep],
        surface.normals[keep],
        surface.areas[keep],
        surface.probe_centers[keep],
        surface.atom_indices[keep],
        surface.buried[keep],
    )


def trim_interface_surface(
    surface: Surface,
    *,
    band: float = 1.5,
    chunk_size: int = 65536,
) -> Surface:
    """Keep buried points that are not near accessible same-molecule points."""
    del chunk_size  # nearest-neighbor trim uses cKDTree directly
    if surface.buried is None:
        raise ValueError("surface must have buried flags before trimming")

    buried_indices = np.flatnonzero(surface.buried)
    if buried_indices.size == 0:
        return surface.empty_like()

    accessible_points = surface.points[~surface.buried]
    buried_points = surface.points[buried_indices]
    if accessible_points.size == 0:
        keep = np.ones(buried_indices.shape[0], dtype=bool)
    else:
        near_accessible = min_distance_to_points(buried_points, accessible_points) <= band
        keep = ~near_accessible

    keep_indices = buried_indices[keep]
    return Surface(
        surface.points[keep_indices],
        surface.normals[keep_indices],
        surface.areas[keep_indices],
        surface.probe_centers[keep_indices],
        surface.atom_indices[keep_indices],
        surface.buried[keep_indices],
    )


def trim_interface_surface_guarded(
    surface: Surface,
    *,
    band: float = 1.5,
    min_trimmed_points: int = 200,
    chunk_size: int = 65536,
) -> Surface:
    """Trim interface surface, reverting to all buried dots if trim is too sparse."""
    trimmed = trim_interface_surface(surface, band=band, chunk_size=chunk_size)
    if trimmed.points.shape[0] >= min_trimmed_points:
        return trimmed
    buried = extract_buried_surface(surface)
    if buried.points.shape[0] > trimmed.points.shape[0]:
        return buried
    return trimmed
