# Bundled benchmark dataset

This folder contains the full apixaban NTF2 protein–ligand benchmark (2,385
structures). Paths in `benchmark.csv` are relative to this directory so examples
and tests do not depend on external filesystem locations.

## Contents

- `pdbs/` — 2,385 gzip-compressed PDB files (`0001.pdb.gz` … `2385.pdb.gz`)
- `benchmark.csv` — Rosetta reference SC for all 2,385 benchmark structures
- `benchmark_pdbs.zip` — compressed archive of `pdbs/` and `benchmark.csv`

Structure filenames are sequential; original pose names are not preserved.
ProDy reads `.pdb.gz` files directly.

## Ligand SMILES

**Apixaban** (all benchmark structures):

```
COC1=CC=C(N2C3=C(C(C(N)=O)=N2)CCN(C4=CC=C(N5CCCCC5=O)C=C4)C3=O)C=C1
```

Extract the zip after clone if `pdbs/` is not present:

```bash
cd testdata && unzip -o benchmark_pdbs.zip
```
