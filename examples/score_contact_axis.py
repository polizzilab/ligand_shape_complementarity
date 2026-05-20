#!/usr/bin/env python
"""Score an interface with the atom-pair contact-axis proxy."""

from __future__ import annotations

import argparse

from shape_complementarity import load_selection, score_contact_axis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdb")
    parser.add_argument("--selection_a", default="protein")
    parser.add_argument("--selection_b", default="hetero and not water")
    parser.add_argument("--contact_shell", type=float, default=0.95)
    parser.add_argument("--max_clash", type=float, default=1.0)
    args = parser.parse_args()

    mol_a = load_selection(args.pdb, args.selection_a)
    mol_b = load_selection(args.pdb, args.selection_b)
    result = score_contact_axis(
        mol_a.xyz,
        mol_a.radii,
        mol_b.xyz,
        mol_b.radii,
        contact_shell=args.contact_shell,
        max_clash=args.max_clash,
    )

    print(f"selection_a_atoms={len(mol_a.radii)}")
    print(f"selection_b_atoms={len(mol_b.radii)}")
    print(f"contact_axis_sc={result.sc:.6f}")
    print(f"contact_axis_distance={result.distance:.6f}")
    print(f"contact_axis_mean_sc={result.mean_sc:.6f}")
    print(f"contact_axis_mean_distance={result.mean_distance:.6f}")
    print(f"n_contacts={result.n_contacts}")
    print(f"contact_shell={result.contact_shell}")
    print(f"max_clash={result.max_clash}")


if __name__ == "__main__":
    main()
