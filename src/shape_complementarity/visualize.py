"""Minimal Plotly views of the surfaces actually used for SC scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from .geometry import nearest_neighbors
from .msms_surface import build_msms_interface_surfaces
from .prody_io import AtomSelection, load_protein_ligand
from .score import (
    InterfaceSurfaces,
    ScoringSurfaces,
    ShapeComplementarityResult,
    build_interface_surfaces,
    resolve_scoring_surfaces,
    score_interface_surfaces,
)


@dataclass(frozen=True)
class ScSurfaceView:
    """Prepared data for a single-structure SC surface visualization."""

    result: ShapeComplementarityResult
    surfaces: InterfaceSurfaces
    scoring: ScoringSurfaces
    protein: AtomSelection
    ligand: AtomSelection
    surface_backend: str


def _surface_points(surface) -> np.ndarray:
    if surface.points.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)
    return surface.points


def _surface_normals(surface) -> np.ndarray:
    if surface.normals.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)
    return surface.normals


def build_sc_surface_view(
    pdb: str | Path,
    *,
    ligand_smiles: str,
    surface_backend: str = "msms",
    probe_radius: float = 1.7,
    density: float = 15.0,
    band: float = 1.5,
    sep: float = 8.0,
    n_sphere_points: int = 96,
    min_trimmed_points: int = 2000,
    min_target_points: int = 2000,
    fallback_nn_dist: float = 6.0,
) -> ScSurfaceView:
    """Build scoring surfaces and resolve the exact subsets used for SC."""
    protein, ligand = load_protein_ligand(pdb, ligand_smiles=ligand_smiles)
    backend = surface_backend.lower()
    msms_surfaces = build_msms_interface_surfaces(
        protein.xyz,
        protein.radii,
        ligand.xyz,
        ligand.radii,
        probe_radius=probe_radius,
        density=density,
        band=band,
        sep=sep,
    )

    if backend == "msms":
        surfaces = msms_surfaces
        backend_label = "msms"
    elif backend == "sphere":
        surfaces = build_interface_surfaces(
            protein.xyz,
            protein.radii,
            ligand.xyz,
            ligand.radii,
            n_sphere_points=n_sphere_points,
            probe_radius=probe_radius,
            band=band,
        )
        backend_label = "sphere"
    elif backend == "hybrid":
        if msms_surfaces.trimmed_a.points.shape[0] < min_trimmed_points:
            surfaces = build_interface_surfaces(
                protein.xyz,
                protein.radii,
                ligand.xyz,
                ligand.radii,
                n_sphere_points=n_sphere_points,
                probe_radius=probe_radius,
                band=band,
            )
            backend_label = "hybrid(sphere fallback)"
        else:
            surfaces = msms_surfaces
            backend_label = "hybrid(msms)"
    else:
        raise ValueError(f"Unknown surface_backend: {surface_backend!r}")

    result = score_interface_surfaces(
        surfaces,
        min_target_points=min_target_points,
        fallback_nn_dist=fallback_nn_dist,
    )
    scoring = resolve_scoring_surfaces(
        surfaces.trimmed_a,
        surfaces.trimmed_b,
        marked_a=surfaces.marked_a,
        marked_b=surfaces.marked_b,
        marked_a_fallback=surfaces.marked_a_full,
        min_target_points=min_target_points,
        fallback_nn_dist=fallback_nn_dist,
    )
    return ScSurfaceView(
        result=result,
        surfaces=surfaces,
        scoring=scoring,
        protein=protein,
        ligand=ligand,
        surface_backend=backend_label,
    )


def _marker_trace(
    name: str,
    points: np.ndarray,
    *,
    color: str,
    size: float,
    opacity: float = 0.9,
    visible: bool | str = True,
) -> go.Scatter3d:
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        name=f"{name} ({len(points)})",
        marker={"size": size, "color": color, "opacity": opacity},
        visible=visible,
    )


def _pair_lines(
    name: str,
    source: np.ndarray,
    target: np.ndarray,
    *,
    color: str,
    max_pairs: int,
    seed: int,
) -> go.Scatter3d:
    if len(source) == 0 or len(target) == 0:
        return go.Scatter3d(x=[], y=[], z=[], mode="lines", name=f"{name} (0)", visible=False)

    nn_dist, nn_idx = nearest_neighbors(source, target)

    rng = np.random.default_rng(seed)
    sample_n = min(max_pairs, len(source))
    sample_idx = rng.choice(len(source), size=sample_n, replace=False)
    x, y, z = [], [], []
    for i in sample_idx:
        a = source[i]
        b = target[nn_idx[i]]
        x.extend([a[0], b[0], None])
        y.extend([a[1], b[1], None])
        z.extend([a[2], b[2], None])
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        name=f"{name} ({sample_n}, med d={np.median(nn_dist[sample_idx]):.2f} A)",
        line={"color": color, "width": 2},
        visible=False,
    )


def _normal_arrows(
    name: str,
    points: np.ndarray,
    normals: np.ndarray,
    *,
    color: str,
    scale: float,
    max_arrows: int,
    seed: int,
) -> go.Scatter3d:
    if len(points) == 0:
        return go.Scatter3d(x=[], y=[], z=[], mode="lines", name=f"{name} (0)", visible=False)
    rng = np.random.default_rng(seed)
    n = min(max_arrows, len(points))
    idx = rng.choice(len(points), size=n, replace=False)
    x, y, z = [], [], []
    for i in idx:
        start = points[i]
        end = start + normals[i] * scale
        x.extend([start[0], end[0], None])
        y.extend([start[1], end[1], None])
        z.extend([start[2], end[2], None])
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        name=name,
        line={"color": color, "width": 2},
        visible=False,
    )


def build_sc_surface_figure(
    view: ScSurfaceView,
    *,
    show_atoms: bool = True,
    show_pair_lines: int = 60,
    show_normals: int = 80,
    normal_scale: float = 0.8,
) -> go.Figure:
    """Build a focused Plotly figure of the four scoring surface sets."""
    fwd_src = _surface_points(view.scoring.forward_source)
    fwd_tgt = _surface_points(view.scoring.forward_target)
    rev_src = _surface_points(view.scoring.reverse_source)
    rev_tgt = _surface_points(view.scoring.reverse_target)

    traces: list[go.Scatter3d] = []
    if show_atoms:
        traces.append(
            _marker_trace("protein atoms", view.protein.xyz, color="#bdbdbd", size=2.0, opacity=0.35)
        )
        traces.append(
            _marker_trace("ligand atoms", view.ligand.xyz, color="#444444", size=3.0, opacity=0.55)
        )

    traces.extend(
        [
            _marker_trace("forward source: protein", fwd_src, color="#0072B2", size=3.5),
            _marker_trace("forward target: ligand", fwd_tgt, color="#E69F00", size=2.5, opacity=0.45),
            _marker_trace("reverse source: ligand", rev_src, color="#CC79A7", size=3.5),
            _marker_trace("reverse target: protein", rev_tgt, color="#009E73", size=2.5, opacity=0.45),
        ]
    )

    if show_pair_lines > 0:
        traces.append(
            _pair_lines(
                "forward NN pairs",
                fwd_src,
                fwd_tgt,
                color="rgba(0,114,178,0.35)",
                max_pairs=show_pair_lines,
                seed=1,
            )
        )
        traces.append(
            _pair_lines(
                "reverse NN pairs",
                rev_src,
                rev_tgt,
                color="rgba(204,121,167,0.35)",
                max_pairs=show_pair_lines,
                seed=2,
            )
        )

    if show_normals > 0:
        traces.append(
            _normal_arrows(
                "protein normals",
                fwd_src,
                _surface_normals(view.scoring.forward_source),
                color="#0072B2",
                scale=normal_scale,
                max_arrows=show_normals,
                seed=3,
            )
        )
        traces.append(
            _normal_arrows(
                "ligand normals",
                rev_src,
                _surface_normals(view.scoring.reverse_source),
                color="#CC79A7",
                scale=normal_scale,
                max_arrows=show_normals,
                seed=4,
            )
        )

    res = view.result
    title = (
        f"[{view.surface_backend}] SC={res.sc:.3f}  fwd={res.forward.sc:.3f}  rev={res.reverse.sc:.3f}  "
        f"| protein trimmed={view.surfaces.trimmed_a.points.shape[0]}  "
        f"ligand trimmed={view.surfaces.trimmed_b.points.shape[0]}"
    )
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene={"aspectmode": "data"},
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )
    return fig


def write_sc_surface_html(
    pdb: str | Path,
    output: str | Path,
    **kwargs,
) -> ScSurfaceView:
    """Build surfaces for a PDB and write a focused Plotly HTML view."""
    view = build_sc_surface_view(pdb, **kwargs)
    fig = build_sc_surface_figure(view, **{k: v for k, v in kwargs.items() if k in {
        "show_atoms", "show_pair_lines", "show_normals", "normal_scale",
    }})
    out = Path(output)
    fig.write_html(str(out), include_plotlyjs="cdn")
    return view
