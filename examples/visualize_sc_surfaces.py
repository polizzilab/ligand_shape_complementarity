#!/usr/bin/env python
"""Write a focused Plotly HTML view of the SC scoring surfaces.

Usage
-----
    python examples/visualize_sc_surfaces.py structure.pdb "COC1=..." \\
        --surface hybrid \\
        --output surfaces.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shape_complementarity.visualize import build_sc_surface_figure, build_sc_surface_view


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdb", help="Input PDB (protein + ligand).")
    parser.add_argument(
        "ligand_smiles",
        help="Ligand SMILES mapped onto the PDB ligand (same as score_msms.py)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output HTML path (default: <pdb_stem>_sc_surfaces.html).",
    )
    parser.add_argument("--probe_radius", type=float, default=1.7)
    parser.add_argument("--density", type=float, default=15.0)
    parser.add_argument("--band", type=float, default=1.5)
    parser.add_argument("--sep", type=float, default=8.0)
    parser.add_argument(
        "--surface",
        choices=("msms", "sphere", "hybrid"),
        default="hybrid",
        help="Surface backend to visualize (default: hybrid).",
    )
    parser.add_argument("--sphere_points", type=int, default=96)
    parser.add_argument("--pair_lines", type=int, default=60, help="Sampled NN pair lines per direction.")
    parser.add_argument("--normals", type=int, default=80, help="Normal arrows per direction (0=off).")
    parser.add_argument("--no_atoms", action="store_true", help="Hide atom coordinates.")
    args = parser.parse_args()

    pdb = Path(args.pdb)
    output = Path(args.output) if args.output else pdb.with_name(f"{pdb.stem}_sc_surfaces.html")

    view = build_sc_surface_view(
        pdb,
        ligand_smiles=args.ligand_smiles,
        surface_backend=args.surface,
        probe_radius=args.probe_radius,
        density=args.density,
        band=args.band,
        sep=args.sep,
        n_sphere_points=args.sphere_points,
    )
    fig = build_sc_surface_figure(
        view,
        show_atoms=not args.no_atoms,
        show_pair_lines=args.pair_lines,
        show_normals=args.normals,
    )
    fig.write_html(str(output), include_plotlyjs="cdn")

    res = view.result
    print(output)
    print(
        f"backend={view.surface_backend}  SC={res.sc:.4f}  fwd={res.forward.sc:.4f}  rev={res.reverse.sc:.4f}  "
        f"protein trimmed={view.surfaces.trimmed_a.points.shape[0]}  "
        f"ligand trimmed={view.surfaces.trimmed_b.points.shape[0]}"
    )
    print(
        "Layers: forward source/target = protein→ligand; "
        "reverse source/target = ligand→protein. "
        "Toggle NN pairs and normals in the legend."
    )


if __name__ == "__main__":
    main()
