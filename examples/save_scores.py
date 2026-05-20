#!/usr/bin/env python
"""Score the hybrid MSMS/sphere dataset with ProcessPool batch parallelism.

Usage
-----
    ~/miniforge3/envs/lasermpnn2/bin/python examples/save_scores.py "COC1=..." \\
        --workers 48
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shape_complementarity.parallel import configure_cpu_threads, pool_kwargs

configure_cpu_threads(1)

EXAMPLES = Path(__file__).parent
PKG_ROOT = EXAMPLES.parent
DEFAULT_CSV_IN = PKG_ROOT / "testdata" / "benchmark.csv"
MIN_TRIMMED_POINTS = 2000


def _resolve_path(path: str, csv_in: Path) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((csv_in.parent / p).resolve())


def _score_one(pdb_path: str, ligand_smiles: str):
    import numpy as np
    import prody as pr

    pr.confProDy(verbosity="none")
    from shape_complementarity.ligand_radii import ligand_radii_from_smiles
    from shape_complementarity.msms_surface import build_msms_interface_surfaces
    from shape_complementarity.radii import sc_atom_radius
    from shape_complementarity.score import build_interface_surfaces, score_interface_surfaces

    try:
        atoms = pr.parsePDB(pdb_path)
        if atoms is None:
            return None
        protein = atoms.select("protein and not hydrogen")
        ligand = atoms.select("hetero and not water and not hydrogen")
        if protein is None or ligand is None:
            return None

        pxyz = np.asarray(protein.getCoords(), dtype=np.float32)
        prad = np.array(
            [
                sc_atom_radius(str(res).strip(), str(name).strip(), str(elem).strip())
                for res, name, elem in zip(protein.getResnames(), protein.getNames(), protein.getElements())
            ],
            dtype=np.float32,
        )
        lxyz = np.asarray(ligand.getCoords(), dtype=np.float32)
        lrad = ligand_radii_from_smiles(ligand_smiles, ligand)

        msms_surfaces = build_msms_interface_surfaces(
            pxyz, prad, lxyz, lrad, probe_radius=1.7, density=15.0, band=1.5, sep=8.0
        )
        msms_res = score_interface_surfaces(msms_surfaces)
        msms_nta = int(msms_surfaces.trimmed_a.points.shape[0])
        used_fallback = msms_nta < MIN_TRIMMED_POINTS

        if used_fallback:
            sphere_surfaces = build_interface_surfaces(
                pxyz, prad, lxyz, lrad, n_sphere_points=96, probe_radius=1.7, band=1.5
            )
            hybrid_res = score_interface_surfaces(sphere_surfaces)
            hybrid_nta = int(sphere_surfaces.trimmed_a.points.shape[0])
        else:
            hybrid_res = msms_res
            hybrid_nta = msms_nta

        return (
            msms_res.sc,
            hybrid_res.sc,
            msms_nta,
            hybrid_nta,
            int(used_fallback),
        )
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "ligand_smiles",
        help="Ligand SMILES mapped onto each PDB for RDKit VdW radii",
    )
    p.add_argument(
        "--csv_in",
        type=Path,
        default=DEFAULT_CSV_IN,
        help="Input CSV with path and Rosetta SC (default: bundled testdata/benchmark.csv)",
    )
    p.add_argument("--path_col", default="path")
    p.add_argument("--rosetta_col", default="shape_comp_AB")
    p.add_argument("--csv_out", type=Path, default=EXAMPLES / "sc_scores_hybrid.csv")
    p.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 1))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_in = args.csv_in.resolve()
    rows = list(csv.DictReader(open(csv_in)))
    paths = [_resolve_path(r[args.path_col], csv_in) for r in rows]
    rosetta = [float(r[args.rosetta_col]) for r in rows]

    print(f"Scoring {len(paths)} structures with {args.workers} workers ...")
    results = [None] * len(paths)
    with ProcessPoolExecutor(**pool_kwargs(max_workers=args.workers)) as pool:
        futures = {
            pool.submit(_score_one, path, args.ligand_smiles): i for i, path in enumerate(paths)
        }
        for fut in tqdm(as_completed(futures), total=len(paths), desc="Scoring", unit="struct"):
            results[futures[fut]] = fut.result()

    with open(args.csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "rosetta_sc", "msms_sc", "hybrid_sc", "msms_nta", "hybrid_nta", "used_sphere_fallback"])
        for i, path in enumerate(paths):
            row = results[i]
            if row is None:
                w.writerow([path, rosetta[i], None, None, None, None, None])
            else:
                w.writerow([path, rosetta[i], *row])

    print(f"Saved {args.csv_out}")


if __name__ == "__main__":
    main()
