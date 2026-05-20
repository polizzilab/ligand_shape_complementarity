#!/usr/bin/env python
"""ProDy + pipeline EDA for a low-SC protein-ligand structure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import prody as pr

from shape_complementarity import extract_buried_surface, trim_interface_surface
from shape_complementarity.geometry import nearest_neighbors
from shape_complementarity.msms_surface import build_msms_interface_surfaces
from shape_complementarity.prody_io import load_protein_ligand
from shape_complementarity.score import resolve_scoring_surfaces, score_interface_surfaces
from shape_complementarity.surface import filter_atoms_by_interface_sep, trim_interface_surface_guarded
from shape_complementarity.visualize import build_sc_surface_figure, build_sc_surface_view


def _directional_diagnostics(source, target, *, weight: float = 0.5, max_nn_dist: float = float("inf")) -> dict:
    if source.points.shape[0] == 0 or target.points.shape[0] == 0:
        return {"n": 0, "med_dist": np.nan, "med_normal_dot": np.nan, "med_sc": np.nan, "mean_sc": np.nan}
    nn_dist, nn_idx = nearest_neighbors(source.points, target.points)
    if max_nn_dist < float("inf"):
        keep = nn_dist <= max_nn_dist
        nn_dist = nn_dist[keep]
        nn_idx = nn_idx[keep]
        if nn_dist.size == 0:
            return {"n": 0, "med_dist": np.nan, "med_normal_dot": np.nan, "med_sc": np.nan, "mean_sc": np.nan}
        src_normals = source.normals[keep]
    else:
        src_normals = source.normals
    tgt_normals = target.normals[nn_idx]
    normal_dot = np.clip(np.sum(src_normals * tgt_normals, axis=1), -1.0, 1.0)
    sc = -normal_dot * np.exp(-(nn_dist * nn_dist) * weight)
    return {
        "n": int(nn_dist.shape[0]),
        "med_dist": float(np.quantile(nn_dist, 0.5)),
        "med_normal_dot": float(np.quantile(normal_dot, 0.5)),
        "med_sc": float(np.quantile(sc, 0.5)),
        "mean_sc": float(sc.mean()),
    }


def _interface_residues(protein, ligand, cutoff: float = 5.0) -> pr.Selection | None:
    contacts = protein.select(f"within {cutoff} of ligand", ligand=ligand)
    if contacts is None:
        return None
    return contacts.select("protein and name CA")


def _print_header(title: str) -> None:
    print(f"\n{'=' * len(title)}\n{title}\n{'=' * len(title)}")


def run_eda(
    pdb: Path,
    *,
    out_dir: Path,
    compare_pdb: Path | None = None,
    contact_cutoff: float = 5.0,
    sep: float = 8.0,
) -> None:
    pr.confProDy(verbosity="none")
    out_dir.mkdir(parents=True, exist_ok=True)

    atoms = pr.parsePDB(str(pdb))
    protein_sel = atoms.select("protein and not hydrogen")
    ligand_sel = atoms.select("hetero and not water and not hydrogen")
    if protein_sel is None or ligand_sel is None:
        raise ValueError("Could not select protein and ligand from PDB")

    _print_header(f"Structure: {pdb.name}")
    print(f"Protein atoms: {protein_sel.numAtoms()}")
    print(f"Ligand atoms:  {ligand_sel.numAtoms()}  resnames={set(ligand_sel.getResnames())}")
    print(f"Protein chains: {set(protein_sel.getChids())}")

    dist_mat = pr.buildDistMatrix(protein_sel, ligand_sel)
    min_dist = float(np.min(dist_mat))
    iface_atom_count = int(np.sum(np.min(dist_mat, axis=1) <= contact_cutoff))
    ca_sel = protein_sel.select("name CA")
    if ca_sel is not None and ca_sel.numAtoms() > 0:
        ca_dist = np.min(pr.buildDistMatrix(ca_sel, ligand_sel), axis=1)
        iface_ca_count = int(np.sum(ca_dist <= contact_cutoff))
    else:
        ca_dist = np.array([])
        iface_ca_count = 0
    print(f"Interface protein atoms within {contact_cutoff} A of ligand: {iface_atom_count}")
    print(f"Interface CAs within {contact_cutoff} A of ligand: {iface_ca_count}")

    if iface_ca_count > 0:
        close_ca_idx = np.where(ca_dist <= contact_cutoff)[0]
        resnums = list(zip(ca_sel.getChids()[close_ca_idx], ca_sel.getResnums()[close_ca_idx], ca_sel.getResnames()[close_ca_idx]))
        print("  " + ", ".join(f"{c}:{r}{n}" for c, r, n in resnums[:30]))
        if len(resnums) > 30:
            print(f"  ... +{len(resnums) - 30} more")

    print(f"Min protein-ligand heavy-atom distance: {min_dist:.2f} A")

    protein, ligand = load_protein_ligand(pdb)
    sep_xyz, sep_rad = filter_atoms_by_interface_sep(protein.xyz, protein.radii, ligand.xyz, sep=sep)
    print(f"Protein atoms within sep={sep} A of ligand (MSMS input): {sep_xyz.shape[0]} / {protein.xyz.shape[0]}")

    surfaces = build_msms_interface_surfaces(
        protein.xyz, protein.radii, ligand.xyz, ligand.radii,
        probe_radius=1.7, density=15.0, band=1.5, sep=sep,
    )
    strict_trim_a = trim_interface_surface(surfaces.marked_a, band=1.5)
    guarded_trim_a = trim_interface_surface_guarded(surfaces.marked_a, band=1.5)
    buried_a = extract_buried_surface(surfaces.marked_a)
    buried_b = extract_buried_surface(surfaces.marked_b)

    _print_header("Surface pipeline counts")
    rows = [
        ("protein MSMS dots", surfaces.surface_a.points.shape[0]),
        ("ligand MSMS dots", surfaces.surface_b.points.shape[0]),
        ("protein buried", buried_a.points.shape[0]),
        ("ligand buried", buried_b.points.shape[0]),
        ("protein strict trim", strict_trim_a.points.shape[0]),
        ("protein guarded trim (used)", guarded_trim_a.points.shape[0]),
        ("ligand trimmed", surfaces.trimmed_b.points.shape[0]),
    ]
    if surfaces.marked_a_full is not None:
        buried_full = extract_buried_surface(surfaces.marked_a_full)
        rows.append(("protein buried (full, no sep filter)", buried_full.points.shape[0]))
    for label, count in rows:
        print(f"  {label:<34} {count:>6}")

    result = score_interface_surfaces(surfaces)
    scoring = resolve_scoring_surfaces(
        surfaces.trimmed_a,
        surfaces.trimmed_b,
        marked_a=surfaces.marked_a,
        marked_b=surfaces.marked_b,
        marked_a_fallback=surfaces.marked_a_full,
    )

    _print_header("Scores")
    print(f"  SC={result.sc:.4f}  forward={result.forward.sc:.4f}  reverse={result.reverse.sc:.4f}")
    print(f"  forward uses {scoring.forward_source.points.shape[0]} src -> {scoring.forward_target.points.shape[0]} tgt")
    print(f"  reverse uses {scoring.reverse_source.points.shape[0]} src -> {scoring.reverse_target.points.shape[0]} tgt")

    fwd_diag = _directional_diagnostics(scoring.forward_source, scoring.forward_target)
    rev_diag = _directional_diagnostics(
        scoring.reverse_source,
        scoring.reverse_target,
        max_nn_dist=scoring.reverse_nn_cap,
    )
    _print_header("Directional diagnostics (median over scored dots)")
    for label, diag in [("forward protein→ligand", fwd_diag), ("reverse ligand→protein", rev_diag)]:
        print(
            f"  {label}: n={diag['n']}  med_dist={diag['med_dist']:.2f} A  "
            f"med_normal_dot={diag['med_normal_dot']:.3f}  med_sc={diag['med_sc']:.3f}  mean_sc={diag['mean_sc']:.3f}"
        )

    # Plot pipeline funnel
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = [r[0] for r in rows[:6]]
    counts = [r[1] for r in rows[:6]]
    axes[0].barh(labels, counts, color=["#0072B2"] * 3 + ["#E69F00"] * 3)
    axes[0].set_title("Surface dot counts")
    axes[0].invert_yaxis()

    dists = []
    names = []
    for label, diag in [("forward", fwd_diag), ("reverse", rev_diag)]:
        if diag["n"] > 0:
            names.append(label)
            dists.append(diag["med_dist"])
    if dists:
        axes[1].bar(names, dists, color=["#0072B2", "#CC79A7"][: len(dists)])
        axes[1].set_ylabel("Median NN distance (A)")
        axes[1].set_title("Scoring geometry")
    fig.tight_layout()
    plot_path = out_dir / f"{pdb.stem}_eda.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved EDA plot → {plot_path}")

    html_path = out_dir / f"{pdb.stem}_sc_surfaces.html"
    view = build_sc_surface_view(pdb)
    build_sc_surface_figure(view).write_html(str(html_path), include_plotlyjs="cdn")
    print(f"Saved surface view → {html_path}")

    if compare_pdb is not None and compare_pdb.exists():
        _print_header(f"Comparison: {compare_pdb.name}")
        cmp_protein, cmp_ligand = load_protein_ligand(compare_pdb)
        cmp_surfaces = build_msms_interface_surfaces(
            cmp_protein.xyz, cmp_protein.radii, cmp_ligand.xyz, cmp_ligand.radii,
            probe_radius=1.7, density=15.0, band=1.5, sep=sep,
        )
        cmp_result = score_interface_surfaces(cmp_surfaces)
        cmp_strict = trim_interface_surface(cmp_surfaces.marked_a, band=1.5).points.shape[0]
        cmp_guard = cmp_surfaces.trimmed_a.points.shape[0]
        print(f"  SC={cmp_result.sc:.4f}  protein strict trim={cmp_strict}  guarded={cmp_guard}")
        print(f"  delta SC vs query: {cmp_result.sc - result.sc:+.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdb")
    parser.add_argument("--out_dir", default=None, help="Output directory for plots/HTML.")
    parser.add_argument(
        "--compare",
        default=None,
        help="Optional second PDB (e.g. same seq from another screen) for side-by-side counts.",
    )
    parser.add_argument("--contact_cutoff", type=float, default=5.0)
    parser.add_argument("--sep", type=float, default=8.0)
    args = parser.parse_args()

    pdb = Path(args.pdb)
    out_dir = Path(args.out_dir) if args.out_dir else pdb.parent / f"{pdb.stem}_eda"
    compare = Path(args.compare) if args.compare else None
    run_eda(pdb, out_dir=out_dir, compare_pdb=compare, contact_cutoff=args.contact_cutoff, sep=args.sep)


if __name__ == "__main__":
    main()
