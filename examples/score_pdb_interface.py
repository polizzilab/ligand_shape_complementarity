#!/usr/bin/env python
"""Score a PDB interface using ProDy selections."""

from __future__ import annotations

import argparse

from shape_complementarity import load_selection, score_complex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdb")
    parser.add_argument(
        "ligand_smiles",
        help="Ligand SMILES mapped onto selection B for RDKit VdW radii",
    )
    parser.add_argument("--selection_a", default="protein")
    parser.add_argument("--selection_b", default="hetero and not water")
    parser.add_argument("--points", type=int, default=96, help="Sphere samples per atom.")
    parser.add_argument("--chunk_size", type=int, default=65536)
    args = parser.parse_args()

    mol_a = load_selection(args.pdb, args.selection_a)
    mol_b = load_selection(args.pdb, args.selection_b, ligand_smiles=args.ligand_smiles)
    result = score_complex(
        mol_a.xyz,
        mol_a.radii,
        mol_b.xyz,
        mol_b.radii,
        n_sphere_points=args.points,
        chunk_size=args.chunk_size,
    )

    print(f"selection_a_atoms={len(mol_a.radii)}")
    print(f"selection_b_atoms={len(mol_b.radii)}")
    print(f"sc={result.sc:.6f}")
    print(f"distance={result.distance:.6f}")
    print(f"area={result.area:.6f}")
    print(f"surface_points={result.n_surface_points_a}+{result.n_surface_points_b}")
    print(f"trimmed_points={result.n_trimmed_points_a}+{result.n_trimmed_points_b}")
    print(f"forward_sc={result.forward.sc:.6f}")
    print(f"reverse_sc={result.reverse.sc:.6f}")


if __name__ == "__main__":
    main()
