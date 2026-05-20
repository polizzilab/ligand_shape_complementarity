"""Atom-pair contact-axis shape-complementarity proxy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import as_float32


@dataclass(frozen=True)
class ContactAxisResult:
    """Summary for the atom-pair contact-axis proxy."""

    sc: float
    distance: float
    mean_sc: float
    mean_distance: float
    n_contacts: int
    contact_shell: float
    max_clash: float


def contact_axis_points(
    atom_xyz_a,
    atom_radii_a,
    atom_xyz_b,
    atom_radii_b,
    *,
    contact_shell: float = 0.95,
    max_clash: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return paired surface points and nonnegative surface gaps for close atom pairs."""
    xyz_a = as_float32(atom_xyz_a)
    radii_a = as_float32(atom_radii_a)
    xyz_b = as_float32(atom_xyz_b)
    radii_b = as_float32(atom_radii_b)

    pair_vectors = xyz_b[None, :, :] - xyz_a[:, None, :]
    center_dist = np.linalg.norm(pair_vectors, axis=-1)
    gap = center_dist - (radii_a[:, None] + radii_b[None, :])
    keep = (gap <= contact_shell) & (gap >= -max_clash) & (center_dist > 1.0e-6)
    if not keep.any():
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, empty, np.empty((0,), dtype=np.float32)

    atom_i, atom_j = np.nonzero(keep)
    unit = pair_vectors[atom_i, atom_j] / center_dist[atom_i, atom_j, None]
    points_a = xyz_a[atom_i] + unit * radii_a[atom_i, None]
    points_b = xyz_b[atom_j] - unit * radii_b[atom_j, None]
    nonnegative_gap = np.clip(gap[atom_i, atom_j], 0.0, None)
    return points_a, points_b, nonnegative_gap


def score_contact_axis(
    atom_xyz_a,
    atom_radii_a,
    atom_xyz_b,
    atom_radii_b,
    *,
    contact_shell: float = 0.95,
    max_clash: float = 1.0,
    weight: float = 0.5,
) -> ContactAxisResult:
    """Score close atom-pair contact axes as perfectly opposed local surfaces."""
    _, _, gaps = contact_axis_points(
        atom_xyz_a,
        atom_radii_a,
        atom_xyz_b,
        atom_radii_b,
        contact_shell=contact_shell,
        max_clash=max_clash,
    )
    if gaps.size == 0:
        return ContactAxisResult(
            sc=float("nan"),
            distance=float("nan"),
            mean_sc=float("nan"),
            mean_distance=float("nan"),
            n_contacts=0,
            contact_shell=contact_shell,
            max_clash=max_clash,
        )

    values = np.exp(-(gaps * gaps) * weight)
    return ContactAxisResult(
        sc=float(np.quantile(values, 0.5)),
        distance=float(np.quantile(gaps, 0.5)),
        mean_sc=float(values.mean()),
        mean_distance=float(gaps.mean()),
        n_contacts=int(gaps.size),
        contact_shell=contact_shell,
        max_clash=max_clash,
    )
