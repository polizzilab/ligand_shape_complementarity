#!/usr/bin/env python
"""Score a protein-ligand interface using MSMS Connolly surfaces.

Usage
-----
    ~/miniforge3/envs/lasermpnn2/bin/python examples/score_msms.py structure.pdb \\
        "COC1=CC=C(N2C3=C(C(C(N)=O)=N2)CCN(C4=CC=C(N5CCCCC5=O)C=C4)C3=O)C=C1" \\
        --protein "protein" \\
        --ligand  "hetero and not water"

The ligand SMILES is mapped onto the PDB ligand geometry with RDKit
``AssignBondOrdersFromTemplate``, and VdW radii are taken from the mapped
RDKit atoms rather than PDB element/name heuristics.

Requirements
------------
The MSMS binary must be on PATH or at
    /programs/x86_64-linux/system/sbgrid_bin/msms  (SBGrid default)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shape_complementarity import load_protein_ligand, score_complex_msms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pdb", help="PDB file to score")
    p.add_argument(
        "ligand_smiles",
        help="Ligand SMILES mapped onto the PDB ligand with AssignBondOrdersFromTemplate",
    )
    p.add_argument("--protein", default="protein", help="ProDy selection for molecule A")
    p.add_argument("--ligand", default="hetero and not water", help="ProDy selection for molecule B")
    p.add_argument("--probe_radius", type=float, default=1.7, help="Probe radius in Å (Rosetta default: 1.7)")
    p.add_argument("--density", type=float, default=15.0, help="MSMS surface density dots/Å² (Rosetta default: 15)")
    p.add_argument("--band", type=float, default=1.5, help="Peripheral band trimming radius in Å")
    p.add_argument("--weight", type=float, default=0.5, help="Distance weight in SC formula")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pdb = args.pdb

    print(f"Loading {pdb} ...")
    protein, ligand = load_protein_ligand(
        pdb,
        protein_selection=args.protein,
        ligand_selection=args.ligand,
        ligand_smiles=args.ligand_smiles,
    )
    print(f"  Protein heavy atoms: {len(protein.xyz)}")
    print(f"  Ligand  heavy atoms: {len(ligand.xyz)}")
    print()

    print("Computing MSMS Connolly surface SC ...")
    result = score_complex_msms(
        protein.xyz,
        protein.radii,
        ligand.xyz,
        ligand.radii,
        probe_radius=args.probe_radius,
        density=args.density,
        band=args.band,
        weight=args.weight,
    )

    print(f"SC (shape complementarity): {result.sc:.4f}")
    print(f"Median distance:            {result.distance:.4f} Å")
    print(f"Interface area:             {result.area:.1f} Ų")

    if args.verbose:
        print()
        print(f"Forward  SC: {result.forward.sc:.4f}  (n={result.forward.n_points} trimmed dots)")
        print(f"Reverse  SC: {result.reverse.sc:.4f}  (n={result.reverse.n_points} trimmed dots)")
        print(f"Total surface pts: protein={result.n_surface_points_a}, ligand={result.n_surface_points_b}")
        print(f"Trimmed pts:       protein={result.n_trimmed_points_a}, ligand={result.n_trimmed_points_b}")


if __name__ == "__main__":
    main()
