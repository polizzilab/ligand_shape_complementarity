# shape-complementarity

Molecular shape-complementarity (SC) scoring with NumPy/SciPy and MSMS.

This package implements a Rosetta-like shape complementarity score for
protein–ligand complexes. It builds Connolly-style interface surfaces (MSMS,
with automatic Fibonacci-sphere fallback for sparse pockets), marks buried
surface dots, trims the interface band, and scores nearest opposing normals
with distance weighting.

**Radii policy (production):**

- **Protein** — Rosetta SC contact radii for the 20 canonical amino acids
- **Ligand** — RDKit periodic-table VdW radii from an RDKit molecule mapped
  onto the PDB ligand with `AssignBondOrdersFromTemplate` (requires `ligand_smiles` positional arg)

## Environment setup

### Option A — conda (recommended)

Create a self-contained environment with MSMS and all Python dependencies:

```bash
cd utils/torch_shape_complementarity
conda env create -f environment.yml
conda activate shape-complementarity
```

Or install into an existing environment:

```bash
conda create -n shape-complementarity -c conda-forge \
  python=3.11 msms numpy scipy prody rdkit tqdm plotly pytest
conda activate shape-complementarity
pip install -e .
```

Verify MSMS is on `PATH`:

```bash
which msms
msms -h    # or run a quick score below
```

The scorer looks for `msms` on `PATH` first. On SBGrid systems it also checks
`/programs/x86_64-linux/system/sbgrid_bin/msms`.

### Option B — use without installing

```bash
export PYTHONPATH=/path/to/torch_shape_complementarity/src:$PYTHONPATH
```

You still need MSMS, RDKit, ProDy, SciPy, and NumPy available in your Python environment.

## Bundled test data

A full apixaban NTF2 benchmark (2,385 structures) ships under `testdata/` so
examples and tests do not depend on external paths:

| Path | Description |
|------|-------------|
| `testdata/pdbs/` | 2,385 gzip-compressed PDBs (`0001.pdb.gz` … `2385.pdb.gz`) |
| `testdata/benchmark.csv` | Rosetta reference SC for all 2,385 structures |
| `testdata/benchmark_pdbs.zip` | Zip archive of `pdbs/` and `benchmark.csv` |

Structure filenames are sequential; ProDy reads `.pdb.gz` directly.

Extract if needed:

```bash
cd testdata && unzip -o benchmark_pdbs.zip
```

**Apixaban SMILES** (CARPdock benchmark):

```
COC1=CC=C(N2C3=C(C(C(N)=O)=N2)CCN(C4=CC=C(N5CCCCC5=O)C=C4)C3=O)C=C1
```

## Score a single structure

```bash
python examples/score_msms.py examples/epic_xtal_ch1.pdb "CC[C@]1(O)C2=C(C(N3CC4=C5[C@@H]([NH3+])CCC6=C5C(N=C4C3=C2)=CC(F)=C6C)=O)COC1=O" -v
```

## Batch scoring

Score the bundled benchmark CSV (2,385 structures; paths relative to `testdata/`):

```bash
python examples/save_scores.py \
  "COC1=CC=C(N2C3=C(C(C(N)=O)=N2)CCN(C4=CC=C(N5CCCCC5=O)C=C4)C3=O)C=C1" \
  --csv_in testdata/benchmark.csv \
  --csv_out examples/sc_scores_benchmark.csv \
  --workers 4
```

For large datasets, increase `--workers`. The script uses a process pool; pair
with `shape_complementarity.parallel.pool_kwargs` so each worker limits BLAS
threads and avoids CPU oversubscription.

## Python API

```python
from shape_complementarity import load_protein_ligand, score_complex_msms

APIX_SMILES = "COC1=CC=C(N2C3=C(C(C(N)=O)=N2)CCN(C4=CC=C(N5CCCCC5=O)C=C4)C3=O)C=C1"

protein, ligand = load_protein_ligand(
    "testdata/pdbs/0001.pdb.gz",
    ligand_smiles=APIX_SMILES,
)

result = score_complex_msms(protein.xyz, protein.radii, ligand.xyz, ligand.radii)
print(result.sc, result.distance, result.area)
```

## Surface visualization

Interactive 3D HTML views of the **exact surface dots used for scoring** — trimmed
interface points, forward/reverse source/target sets, optional nearest-neighbor
pair lines, and surface normals. Requires [Plotly](https://plotly.com/python/).

Install Plotly if needed:

```bash
conda install -c conda-forge plotly
# or: pip install plotly
```

Generate a view with the production hybrid backend (MSMS + sphere fallback):

```bash
python examples/visualize_sc_surfaces.py \
  testdata/pdbs/0001.pdb.gz \
  "COC1=CC=C(N2C3=C(C(C(N)=O)=N2)CCN(C4=CC=C(N5CCCCC5=O)C=C4)C3=O)C=C1" \
  --surface hybrid \
  --output 0001_sc_surfaces.html
```

Open the HTML file in a browser. Use the legend to toggle layers:

| Layer | Meaning |
|-------|---------|
| **forward source** (blue) | Protein trimmed dots scored in the forward direction |
| **forward target** (orange) | Ligand dots matched against protein |
| **reverse source** (pink) | Ligand trimmed dots scored in the reverse direction |
| **reverse target** (green) | Protein dots matched against ligand |
| **forward/reverse NN pairs** | Sampled nearest-neighbor lines between source and target |
| **protein/ligand normals** | Sampled surface normal arrows |

Useful options:

- `--surface msms` — MSMS only (no sphere fallback)
- `--surface sphere` — Fibonacci sphere approximation only
- `--pair_lines 0` — hide NN pair lines
- `--normals 0` — hide normal arrows
- `--no_atoms` — surface dots only

```bash
python examples/visualize_sc_surfaces.py structure.pdb "..." \
  --surface msms \
  --pair_lines 0 \
  --normals 0 \
  --no_atoms
```

Inspect a sparse-pocket outlier (triggers sphere fallback in hybrid mode):

```bash
python examples/visualize_sc_surfaces.py \
  testdata/pdbs/0027.pdb.gz \
  "COC1=CC=C(N2C3=C(C(C(N)=O)=N2)CCN(C4=CC=C(N5CCCCC5=O)C=C4)C3=O)C=C1" \
  --surface hybrid
```

Python API:

```python
from shape_complementarity.visualize import build_sc_surface_view, build_sc_surface_figure

view = build_sc_surface_view(
    "testdata/pdbs/0001.pdb.gz",
    ligand_smiles=APIX_SMILES,
    surface_backend="hybrid",
)
fig = build_sc_surface_figure(view, show_pair_lines=60, show_normals=80)
fig.write_html("surfaces.html", include_plotlyjs="cdn")
```

## Tests

```bash
pytest tests/ -q
```

## Correlation benchmark

Re-score the bundled 2,385-structure dataset and plot correlation against Rosetta:

```bash
python examples/save_scores.py \
  "COC1=CC=C(N2C3=C(C(C(N)=O)=N2)CCN(C4=CC=C(N5CCCCC5=O)C=C4)C3=O)C=C1" \
  --csv_in testdata/benchmark.csv \
  --csv_out examples/sc_scores_hybrid.csv \
  --workers 48

python examples/plot_correlation_from_csv.py \
  --csv examples/sc_scores_hybrid.csv \
  --method "Hybrid (AA Rosetta + SMILES RDKit)"
```

On this dataset Pearson **r ≈ 0.94** vs Rosetta SC. For an external CSV with
absolute PDB paths, pass `--csv_in /path/to/merged_data.csv` instead.

## Notes

- MSMS is the primary surface backend; sphere fallback activates when the
  trimmed protein MSMS surface has fewer than 2,000 dots.
- Default MSMS parameters match Rosetta: `probe_radius=1.7`, `density=15`.
- `ligand_smiles` is a **required positional argument** for CLI scoring; there is no default
  because the ligand differs for every run.
