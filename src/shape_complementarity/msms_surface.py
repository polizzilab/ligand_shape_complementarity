"""MSMS-backed Connolly surface generation."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .score import (
    InterfaceSurfaces,
    ShapeComplementarityResult,
    build_interface_surfaces,
    score_interface_surfaces,
)
from .surface import (
    Surface,
    extract_buried_surface,
    filter_atoms_by_interface_sep,
    mark_buried_by_atoms,
    trim_interface_surface,
    trim_interface_surface_guarded,
)

MSMS_BINARY = "/programs/x86_64-linux/system/sbgrid_bin/msms"


def _find_msms() -> str:
    """Return the msms executable path, preferring the SBGrid install."""
    if Path(MSMS_BINARY).is_file():
        return MSMS_BINARY
    found = shutil.which("msms")
    if found:
        return found
    raise RuntimeError(
        f"msms binary not found at {MSMS_BINARY} and not on PATH. "
        "Install via SBGrid or set MSMS_BINARY."
    )


def _empty_surface() -> Surface:
    empty3 = np.empty((0, 3), dtype=np.float32)
    empty_idx = np.empty((0,), dtype=np.int64)
    return Surface(empty3, empty3, empty3[:0], empty3, empty_idx)


def _write_xyzr(path: Path, xyz: np.ndarray, radii: np.ndarray) -> None:
    """Write an MSMS .xyzr input file (x y z r, one atom per line)."""
    with open(path, "w") as fh:
        for (x, y, z), r in zip(xyz, radii):
            fh.write(f"{x:.4f} {y:.4f} {z:.4f} {r:.4f}\n")


def _parse_vert(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse an MSMS .vert file."""
    coords, normals, atom_idx = [], [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
                normals.append([float(parts[3]), float(parts[4]), float(parts[5])])
                atom_idx.append(int(parts[7]) - 1)
            except ValueError:
                continue
    return (
        np.array(coords, dtype=np.float32),
        np.array(normals, dtype=np.float32),
        np.array(atom_idx, dtype=np.int64),
    )


def _parse_face(path: Path) -> np.ndarray:
    """Parse an MSMS .face file."""
    faces = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                faces.append([int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2]) - 1])
            except ValueError:
                continue
    return np.array(faces, dtype=np.int64)


def _compute_vertex_areas(coords: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Distribute triangle areas equally among the three vertices."""
    areas = np.zeros(len(coords), dtype=np.float32)
    if len(faces) == 0:
        return areas

    v0 = coords[faces[:, 0]]
    v1 = coords[faces[:, 1]]
    v2 = coords[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    tri_area = 0.5 * np.linalg.norm(cross, axis=1)

    np.add.at(areas, faces[:, 0], tri_area / 3.0)
    np.add.at(areas, faces[:, 1], tri_area / 3.0)
    np.add.at(areas, faces[:, 2], tri_area / 3.0)
    return areas


def build_msms_surface(
    atom_xyz,
    atom_radii,
    *,
    probe_radius: float = 1.4,
    density: float = 3.0,
    msms_binary: str | None = None,
) -> Surface:
    """Build a Connolly solvent-excluded surface using MSMS."""
    binary = msms_binary or _find_msms()
    xyz = np.asarray(atom_xyz, dtype=np.float32)
    radii = np.asarray(atom_radii, dtype=np.float32)

    if xyz.shape[0] == 0:
        return _empty_surface()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        xyzr_path = tmp / "mol.xyzr"
        out_prefix = str(tmp / "mol")

        _write_xyzr(xyzr_path, xyz, radii)

        cmd = [
            binary,
            "-if",
            str(xyzr_path),
            "-of",
            out_prefix,
            "-probe_radius",
            str(probe_radius),
            "-density",
            str(density),
            "-no_header",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"MSMS failed (exit {result.returncode}):\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )

        vert_path = tmp / "mol.vert"
        face_path = tmp / "mol.face"

        if not vert_path.exists():
            raise RuntimeError(
                f"MSMS produced no .vert file.\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )

        coords, normals, atom_idx = _parse_vert(vert_path)

        if face_path.exists():
            faces = _parse_face(face_path)
            areas = _compute_vertex_areas(coords, faces)
        else:
            n = len(coords)
            areas = np.full(n, 1.0 / max(n, 1), dtype=np.float32)

    if len(coords) == 0:
        return _empty_surface()

    probe_centers = coords + normals * probe_radius
    return Surface(
        points=coords,
        normals=normals,
        areas=areas,
        probe_centers=probe_centers,
        atom_indices=atom_idx,
    )


def _needs_full_protein_fallback(
    marked_a: Surface,
    trimmed_a: Surface,
    *,
    min_target_points: int,
) -> bool:
    if trimmed_a.points.shape[0] >= min_target_points:
        return False
    if marked_a.buried is None:
        return True
    return int(marked_a.buried.sum()) < min_target_points


def build_msms_interface_surfaces(
    atom_xyz_a,
    atom_radii_a,
    atom_xyz_b,
    atom_radii_b,
    *,
    probe_radius: float = 1.4,
    density: float = 3.0,
    band: float = 1.5,
    sep: float = 8.0,
    sep_b: float | None = None,
    min_target_points: int = 2000,
    msms_binary: str | None = None,
    chunk_size: int = 65536,
) -> InterfaceSurfaces:
    """Build MSMS-based interface surfaces for two molecules."""
    xyz_a = np.asarray(atom_xyz_a, dtype=np.float32)
    rad_a = np.asarray(atom_radii_a, dtype=np.float32)
    xyz_b = np.asarray(atom_xyz_b, dtype=np.float32)
    rad_b = np.asarray(atom_radii_b, dtype=np.float32)

    if sep > 0:
        surf_xyz_a, surf_rad_a = filter_atoms_by_interface_sep(xyz_a, rad_a, xyz_b, sep=sep)
    else:
        surf_xyz_a, surf_rad_a = xyz_a, rad_a

    if sep_b is not None and sep_b > 0:
        surf_xyz_b, surf_rad_b = filter_atoms_by_interface_sep(xyz_b, rad_b, xyz_a, sep=sep_b)
    else:
        surf_xyz_b, surf_rad_b = xyz_b, rad_b

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(
            build_msms_surface,
            surf_xyz_a,
            surf_rad_a,
            probe_radius=probe_radius,
            density=density,
            msms_binary=msms_binary,
        )
        fut_b = pool.submit(
            build_msms_surface,
            surf_xyz_b,
            surf_rad_b,
            probe_radius=probe_radius,
            density=density,
            msms_binary=msms_binary,
        )
        surface_a = fut_a.result()
        surface_b = fut_b.result()
    marked_a = mark_buried_by_atoms(
        surface_a,
        xyz_b,
        rad_b,
        probe_radius=probe_radius,
        chunk_size=chunk_size,
    )
    marked_b = mark_buried_by_atoms(
        surface_b,
        xyz_a,
        rad_a,
        probe_radius=probe_radius,
        chunk_size=chunk_size,
    )
    trimmed_a = trim_interface_surface_guarded(marked_a, band=band, chunk_size=chunk_size)
    trimmed_b = trim_interface_surface(marked_b, band=band, chunk_size=chunk_size)

    marked_a_full = None
    if sep > 0 and _needs_full_protein_fallback(marked_a, trimmed_a, min_target_points=min_target_points):
        surface_a_full = build_msms_surface(
            xyz_a,
            rad_a,
            probe_radius=probe_radius,
            density=density,
            msms_binary=msms_binary,
        )
        marked_a_full = mark_buried_by_atoms(
            surface_a_full,
            xyz_b,
            rad_b,
            probe_radius=probe_radius,
            chunk_size=chunk_size,
        )

    return InterfaceSurfaces(
        surface_a=surface_a,
        surface_b=surface_b,
        marked_a=marked_a,
        marked_b=marked_b,
        trimmed_a=trimmed_a,
        trimmed_b=trimmed_b,
        marked_a_full=marked_a_full,
    )


def score_msms_with_sphere_fallback(
    atom_xyz_a,
    atom_radii_a,
    atom_xyz_b,
    atom_radii_b,
    *,
    probe_radius: float = 1.4,
    density: float = 3.0,
    band: float = 1.5,
    sep: float = 8.0,
    sep_b: float | None = None,
    n_sphere_points: int = 96,
    min_trimmed_points: int = 2000,
    weight: float = 0.5,
    min_target_points: int = 2000,
    fallback_nn_dist: float = 6.0,
    msms_binary: str | None = None,
    chunk_size: int = 65536,
) -> ShapeComplementarityResult:
    """Score with MSMS surfaces, falling back to sphere surfaces when sparse."""
    msms_surfaces = build_msms_interface_surfaces(
        atom_xyz_a,
        atom_radii_a,
        atom_xyz_b,
        atom_radii_b,
        probe_radius=probe_radius,
        density=density,
        band=band,
        sep=sep,
        sep_b=sep_b,
        min_target_points=min_target_points,
        msms_binary=msms_binary,
        chunk_size=chunk_size,
    )
    if msms_surfaces.trimmed_a.points.shape[0] >= min_trimmed_points:
        return score_interface_surfaces(
            msms_surfaces,
            weight=weight,
            min_target_points=min_target_points,
            fallback_nn_dist=fallback_nn_dist,
            chunk_size=chunk_size,
        )

    sphere_surfaces = build_interface_surfaces(
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
        sphere_surfaces,
        weight=weight,
        min_target_points=min_target_points,
        fallback_nn_dist=fallback_nn_dist,
        chunk_size=chunk_size,
    )


def score_complex_msms(
    atom_xyz_a,
    atom_radii_a,
    atom_xyz_b,
    atom_radii_b,
    *,
    probe_radius: float = 1.4,
    density: float = 3.0,
    band: float = 1.5,
    sep: float = 8.0,
    sep_b: float | None = None,
    weight: float = 0.5,
    min_target_points: int = 2000,
    fallback_nn_dist: float = 6.0,
    msms_binary: str | None = None,
    chunk_size: int = 65536,
) -> ShapeComplementarityResult:
    """Compute SC using MSMS Connolly surfaces."""
    return score_msms_with_sphere_fallback(
        atom_xyz_a,
        atom_radii_a,
        atom_xyz_b,
        atom_radii_b,
        probe_radius=probe_radius,
        density=density,
        band=band,
        sep=sep,
        sep_b=sep_b,
        weight=weight,
        min_target_points=min_target_points,
        fallback_nn_dist=fallback_nn_dist,
        msms_binary=msms_binary,
        chunk_size=chunk_size,
    )
