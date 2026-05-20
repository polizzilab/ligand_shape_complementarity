#!/usr/bin/env python
"""Write an interactive Plotly HTML view of approximate interface surfaces."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from shape_complementarity import build_interface_surfaces, contact_axis_points, load_selection, score_surfaces
from shape_complementarity.geometry import nearest_neighbors


def _downsample(array: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if array.shape[0] <= max_points:
        return array
    rng = np.random.default_rng(seed)
    indices = rng.choice(array.shape[0], size=max_points, replace=False)
    return array[indices]


def _scatter_points(name: str, points: np.ndarray, color: str, size: float, opacity: float = 0.8) -> go.Scatter3d:
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        name=f"{name} ({len(points)})",
        marker={"size": size, "color": color, "opacity": opacity},
    )


def _atom_trace(name: str, xyz: np.ndarray, color: str) -> go.Scatter3d:
    return _scatter_points(name, xyz, color=color, size=4.0, opacity=0.55)


def _normal_trace(name: str, points: np.ndarray, normals: np.ndarray, color: str, scale: float) -> go.Scatter3d:
    x, y, z = [], [], []
    for point, normal in zip(points, normals, strict=False):
        end = point + normal * scale
        x.extend([point[0], end[0], None])
        y.extend([point[1], end[1], None])
        z.extend([point[2], end[2], None])
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        name=name,
        line={"color": color, "width": 2},
        opacity=0.7,
    )


def _nearest_link_trace(points_a: np.ndarray, points_b: np.ndarray, max_links: int, seed: int) -> go.Scatter3d:
    if len(points_a) == 0 or len(points_b) == 0:
        return go.Scatter3d(x=[], y=[], z=[], mode="lines", name="nearest links (0)")
    rng = np.random.default_rng(seed)
    sample_size = min(max_links, len(points_a))
    source_indices = rng.choice(len(points_a), size=sample_size, replace=False)
    sampled_a = points_a[source_indices]
    _, nn_idx = nearest_neighbors(sampled_a, points_b)
    sampled_b = points_b[nn_idx]
    x, y, z = [], [], []
    for a, b in zip(sampled_a, sampled_b, strict=False):
        x.extend([a[0], b[0], None])
        y.extend([a[1], b[1], None])
        z.extend([a[2], b[2], None])
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        name=f"nearest links ({sample_size})",
        line={"color": "rgba(40,40,40,0.35)", "width": 2},
    )


def _contact_axis_trace(points_a: np.ndarray, points_b: np.ndarray, max_links: int, seed: int) -> go.Scatter3d:
    if len(points_a) == 0:
        return go.Scatter3d(x=[], y=[], z=[], mode="lines", name="contact axes (0)")
    rng = np.random.default_rng(seed)
    sample_size = min(max_links, len(points_a))
    indices = rng.choice(len(points_a), size=sample_size, replace=False)
    x, y, z = [], [], []
    for a, b in zip(points_a[indices], points_b[indices], strict=False):
        x.extend([a[0], b[0], None])
        y.extend([a[1], b[1], None])
        z.extend([a[2], b[2], None])
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        name=f"contact axes ({sample_size})",
        line={"color": "rgba(80,0,120,0.45)", "width": 3},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdb")
    parser.add_argument("--selection_a", default="protein")
    parser.add_argument("--selection_b", default="hetero and not water")
    parser.add_argument("--points", type=int, default=512, help="Sphere samples per atom.")
    parser.add_argument("--band", type=float, default=1.5)
    parser.add_argument("--output", default="surface_debug.html")
    parser.add_argument("--max_surface_points", type=int, default=12000)
    parser.add_argument("--max_normal_arrows", type=int, default=400)
    parser.add_argument("--max_links", type=int, default=500)
    parser.add_argument("--contact_shell", type=float, default=0.95)
    args = parser.parse_args()

    mol_a = load_selection(args.pdb, args.selection_a)
    mol_b = load_selection(args.pdb, args.selection_b)
    surfaces = build_interface_surfaces(
        mol_a.xyz,
        mol_a.radii,
        mol_b.xyz,
        mol_b.radii,
        n_sphere_points=args.points,
        band=args.band,
    )
    result = score_surfaces(surfaces.trimmed_a, surfaces.trimmed_b)

    marked_a_points = surfaces.marked_a.points
    marked_b_points = surfaces.marked_b.points
    buried_a = surfaces.marked_a.buried
    buried_b = surfaces.marked_b.buried
    trimmed_a_points = surfaces.trimmed_a.points
    trimmed_b_points = surfaces.trimmed_b.points
    trimmed_a_normals = surfaces.trimmed_a.normals
    trimmed_b_normals = surfaces.trimmed_b.normals

    traces = [
        _atom_trace("A atoms", mol_a.xyz, "#1f77b4"),
        _atom_trace("B atoms", mol_b.xyz, "#d62728"),
        _scatter_points(
            "A accessible surface",
            _downsample(marked_a_points[~buried_a], args.max_surface_points, 1),
            "#9ecae1",
            1.5,
            0.25,
        ),
        _scatter_points(
            "B accessible surface",
            _downsample(marked_b_points[~buried_b], args.max_surface_points, 2),
            "#fcbba1",
            1.5,
            0.25,
        ),
        _scatter_points(
            "A buried surface",
            _downsample(marked_a_points[buried_a], args.max_surface_points, 3),
            "#08519c",
            2.0,
            0.55,
        ),
        _scatter_points(
            "B buried surface",
            _downsample(marked_b_points[buried_b], args.max_surface_points, 4),
            "#a50f15",
            2.0,
            0.55,
        ),
        _scatter_points("A trimmed interface", trimmed_a_points, "#00ccff", 3.0, 0.95),
        _scatter_points("B trimmed interface", trimmed_b_points, "#ff7f0e", 3.0, 0.95),
        _nearest_link_trace(trimmed_a_points, trimmed_b_points, args.max_links, 5),
    ]
    contact_a, contact_b, _ = contact_axis_points(
        mol_a.xyz,
        mol_a.radii,
        mol_b.xyz,
        mol_b.radii,
        contact_shell=args.contact_shell,
    )
    traces.append(_contact_axis_trace(contact_a, contact_b, args.max_links, 8))

    if len(trimmed_a_points):
        normal_idx = _downsample(np.arange(len(trimmed_a_points))[:, None], args.max_normal_arrows, 6).ravel()
        traces.append(_normal_trace("A trimmed normals", trimmed_a_points[normal_idx], trimmed_a_normals[normal_idx], "#0066ff", 0.7))
    if len(trimmed_b_points):
        normal_idx = _downsample(np.arange(len(trimmed_b_points))[:, None], args.max_normal_arrows, 7).ravel()
        traces.append(_normal_trace("B trimmed normals", trimmed_b_points[normal_idx], trimmed_b_normals[normal_idx], "#ff3300", 0.7))

    title = (
        f"Approx SC={result.sc:.3f}, dist={result.distance:.3f}, area={result.area:.1f}; "
        f"trimmed={len(trimmed_a_points)}+{len(trimmed_b_points)}"
    )
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene={"aspectmode": "data"},
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )

    output = Path(args.output)
    fig.write_html(str(output), include_plotlyjs="cdn")
    print(output)
    print(title)


if __name__ == "__main__":
    main()
