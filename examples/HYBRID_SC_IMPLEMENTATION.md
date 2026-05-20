# Hybrid Shape Complementarity — Implementation Summary

This document describes the **current recommended SC pipeline** in
`utils/torch_shape_complementarity/` so another agent can reproduce behavior and
findings **without reading or copying Rosetta source code**.

Reference benchmark: **2,385 protein–ligand PDBs** from the CARPdock apixaban NTF2
screen(s), with Rosetta SC values used only as an external validation target.

---

## 1. Goal

Compute **protein–ligand shape complementarity (SC)** in [0, 1] that:

- Correlates with Rosetta `ShapeComplementarity` scores on the same PDBs
- Runs on GPU-friendly PyTorch tensors
- Handles ~5% of structures where **MSMS** under-represents the binding-pocket surface

**Recommended entry point:** `score_msms_with_sphere_fallback()` (also exposed as
`score_complex_msms()`).

---

## 2. Key finding (why hybrid exists)

### MSMS alone is good most of the time

On the full test set (fixed MSMS pipeline):

| Metric vs Rosetta | MSMS only |
|-------------------|-----------|
| Pearson r         | 0.588     |
| Spearman r        | 0.947     |
| Gross failures*   | 42        |

\*Gross failure = Rosetta SC > 0.5 and our SC < 0.4

MSMS preserves **ranking** well but catastrophically fails on a subset of pockets
where the ligand looks visually buried but MSMS assigns very few protein
interface dots.

### Sphere approximation alone is not sufficient

Fibonacci-sphere contact surfaces (no MSMS) fix gross failures but **destroy global
ranking** (Spearman ~0.58 vs Rosetta).

### Hybrid wins

| Metric vs Rosetta | Hybrid (MSMS + sphere fallback) |
|-------------------|----------------------------------|
| Pearson r         | **0.969**                        |
| Spearman r        | **0.971**                        |
| RMSE              | **0.016**                        |
| Gross failures    | **0**                            |

Sphere fallback triggers on **129 / 2,385 structures (5.4%)** when MSMS protein
trimmed dots < 2000.

**Canonical example:** `seq_0389` (screen1)

| Method  | SC    | Protein trimmed dots | Rosetta |
|---------|-------|----------------------|---------|
| MSMS    | 0.145 | 301                  | 0.541   |
| Hybrid  | 0.624 | 172 (sphere)         | 0.541   |
| seq_0215 (control) | 0.782 | 7880 (MSMS) | 0.786 |

Visual comparison HTML files exist under
`examples/outliers_manual_inspection/still_low/seq_0389_eda/`.

---

## 3. Clean-room algorithm overview

This is an **independent implementation** inspired by published shape
complementarity ideas (Lawrence & Colman 1993; solvent-excluded surfaces). It
does **not** port Rosetta's Connolly C++ code.

### Molecule setup (protein = A, ligand = B)

- Load with ProDy: `protein and not hydrogen`, `hetero and not water and not hydrogen`
- **Hydrogens are always stripped** (matches Rosetta SC practice of using heavy
  atoms only)
- Use **SC contact radii** from `rosetta_radii.py` (numeric values parsed from
  the public `sc_radii.lib` data file — not Rosetta code)

### Surface generation (two backends)

#### Backend A: MSMS (default)

1. **Protein atom filter (`sep=8 Å`):** Only protein heavy atoms with any
   ligand heavy atom within 8 Å are passed to MSMS for protein surface generation.
   Ligand uses all heavy atoms. Burial marking still uses **full** atom sets.
2. Run **MSMS** binary (`/programs/x86_64-linux/system/sbgrid_bin/msms`) at
   `probe_radius=1.7`, `density=15.0` dots/Å².
3. Optionally build `marked_a_full`: full-protein MSMS surface for buried fallback.

#### Backend B: Fibonacci sphere approximation (fallback)

1. Place ~96 uniformly distributed points on each atom's van der Waals sphere.
2. Keep points whose solvent probe center (point + normal × probe_radius) is not
   inside any other same-molecule expanded sphere (accessibility test).
3. Uses **all** protein/ligand heavy atoms (no `sep=8` filter on surface build).

### Interface marking (`mark_buried_by_atoms`)

For each surface dot, place a solvent probe center along the outward normal.
Mark dot as **buried-by-opponent** if that probe center lies inside any opponent
atom sphere expanded by `probe_radius`.

> **Naming note:** `buried=True` means **occluded by the other molecule's probe**,
> not "buried in the protein core." Pocket walls are included in the surface.

### Interface trimming (`trim_interface_surface`)

Among dots marked buried-by-opponent, **remove rim dots**: those within `band=1.5 Å`
of a same-molecule dot that is **not** buried-by-opponent (peripheral band trim).

**Protein-only trim guard** (`trim_interface_surface_guarded`): if strict trim
leaves < 200 protein dots, revert to **all** buried-by-opponent protein dots
(no rim removal).

### Hybrid selection (`score_msms_with_sphere_fallback`)

```
msms_surfaces = build_msms_interface_surfaces(...)
if msms_surfaces.trimmed_a.n_points >= min_trimmed_points (default 2000):
    score msms_surfaces
else:
    sphere_surfaces = build_interface_surfaces(...)  # Fibonacci backend
    score sphere_surfaces
```

Both paths then use the **same scoring logic** below.

---

## 4. Scoring logic (`score_interface_surfaces` → `score_surfaces`)

### Directional scoring

Two directions, symmetric:

- **Forward:** protein trimmed → ligand target
- **Reverse:** ligand source → protein target

For each source dot, find **nearest** target dot. Per-pair score:

```
normal_dot = clamp(source_normal · target_normal, -1, 1)
pair_sc    = -normal_dot * exp(-distance² * weight)    # weight default 0.5
```

Directional SC = **median** of pair scores (source dots with no in-range target
excluded when a distance cap applies).

Final SC:

```
SC = max(0, (forward_median + reverse_median) / 2)
```

### Scoring-time fallbacks (apply to both MSMS and sphere paths)

#### Target fallback (`_target_surface_for_direction`)

If a direction's trimmed target has < `min_target_points` (2000):

1. Fall back to all **buried-by-opponent** dots on that molecule
2. If still sparse, use `marked_a_full` buried dots (protein only, from full MSMS)
3. Restrict fallback target to dots within `fallback_nn_dist=6.0 Å` of opposing
   trimmed source

#### Reverse source matching (`_interface_matched_source`)

When protein trimmed is sparse **or** ligand trimmed > protein trimmed:

- Reverse source = unique ligand dots that are nearest neighbors of at least one
  protein target dot (deduplicated)
- Prevents thousands of off-interface ligand dots from collapsing reverse median → 0

#### Reverse distance cap

When expanded protein target is used (`target_a` larger than trimmed protein
source), exclude reverse pairs with nearest distance > 6 Å.

---

## 5. Default parameters (production)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `probe_radius` | 1.7 Å | Solvent probe |
| `density` | 15.0 | MSMS dots/Å² |
| `band` | 1.5 Å | Peripheral trim width |
| `sep` | 8.0 Å | Protein atoms for MSMS input |
| `min_trimmed_points` | 2000 | Hybrid sphere fallback threshold |
| `min_target_points` | 2000 | Target surface fallback threshold |
| `fallback_nn_dist` | 6.0 Å | Spatial cap for fallback targets / reverse |
| `n_sphere_points` | 96 | Sphere points per atom (fallback backend) |
| `weight` | 0.5 | Distance exponential in pair score |

---

## 6. Code map

```
src/shape_complementarity/
├── msms_surface.py       # MSMS build, score_msms_with_sphere_fallback, score_complex_msms
├── surface.py            # Surface dataclass, burial, trim, trim guard, sep filter
├── score.py              # Directional scoring, all scoring-time fallbacks
├── rosetta_radii.py      # SC contact radii lookup (data-driven)
├── prody_io.py           # PDB loading, H stripped
├── visualize.py          # Plotly HTML (--surface msms|sphere|hybrid)
└── __init__.py           # Public exports

examples/
├── save_scores.py                    # Batch MSMS rescore (older; use hybrid for production)
├── visualize_sc_surfaces.py          # Surface HTML viewer
├── eda_sc_structure.py               # ProDy + pipeline diagnostics
├── sc_scores.csv                     # Original MSMS benchmark (pre-hybrid era)
├── sc_scores_sphere.csv              # Pure sphere benchmark
├── sc_scores_hybrid.csv              # **Hybrid benchmark (authoritative)**
└── HYBRID_SC_IMPLEMENTATION.md       # This file
```

### Primary API

```python
from shape_complementarity import (
    load_protein_ligand,
    score_msms_with_sphere_fallback,
)

protein, ligand = load_protein_ligand("complex.pdb", use_rosetta_radii=True)

result = score_msms_with_sphere_fallback(
    protein.xyz, protein.radii,
    ligand.xyz,  ligand.radii,
    probe_radius=1.7,
    density=15.0,
    band=1.5,
    sep=8.0,
    min_trimmed_points=2000,
)

print(result.sc)                  # final SC
print(result.forward.sc)          # protein → ligand
print(result.reverse.sc)          # ligand → protein
print(result.n_trimmed_points_a)    # protein trimmed dot count
```

`score_complex_msms(...)` is a thin wrapper around the same function.

---

## 7. How to run

### Environment

```bash
conda activate lasermpnn2
cd utils/torch_shape_complementarity
export PYTHONPATH=src
```

Requires: `torch`, `prody`, `numpy`, MSMS on PATH (SBGrid default).

### Score one PDB

```bash
PYTHONPATH=src python examples/score_msms.py structure.pdb -v
```

(`score_msms.py` calls `score_complex_msms` → hybrid.)

### Visualize surfaces

```bash
PYTHONPATH=src python examples/visualize_sc_surfaces.py structure.pdb \
  --surface hybrid \
  --output out.html
```

Layers in HTML:

- Blue = forward protein source
- Orange = forward ligand target
- Pink = reverse ligand source
- Green = reverse protein target

### EDA on a problematic structure

```bash
PYTHONPATH=src python examples/eda_sc_structure.py structure.pdb \
  --out-dir structure_eda/
```

---

## 8. Benchmark artifacts

| File | Contents |
|------|----------|
| `examples/sc_scores_hybrid.csv` | path, rosetta_sc, msms_sc, hybrid_sc, nta, fallback flag |
| `examples/sc_scores_sphere.csv` | Pure sphere comparison |
| `examples/sc_scores.csv` | Older MSMS-only run (stale for outliers) |

Regenerate hybrid benchmark (48 workers, ~37 min):

```python
# Pattern: for each PDB in sc_scores.csv
#   msms = build_msms_interface_surfaces + score_interface_surfaces
#   hybrid = score_msms_with_sphere_fallback
```

---

## 9. Diagnostics that matter

When SC looks wrong but the pose looks fine:

1. **Count protein trimmed dots** — if << 2000, hybrid should switch to sphere
2. **Compare MSMS vs sphere SC** on the same PDB — large gap ⇒ MSMS geometry issue
3. **Check forward vs reverse** — reverse ≈ 0 often meant unmatched ligand surface
   before reverse-source matching fix
4. **ProDy distance matrix** — heavy-atom contacts within 5 Å confirm visual contact
5. **Do not confuse** atom–atom distance with probe-overlap burial — many close atoms
   can have zero MSMS dots on that atom

Duplicate seq IDs across screens can have **wildly different** MSMS dot counts
(same sequence, different docked pose).

---

## 10. What this implementation deliberately is NOT

- **Not** a copy of Rosetta's Connolly surface C++ (`MolecularSurfaceCalculator`)
- **Not** using Rosetta executables at runtime (optional validation only)
- **Not** protein–protein validated; pipeline assumes **protein = A, ligand = B**
- **Not** using hydrogens anywhere in SC
- MSMS ≠ Rosetta Connolly; they agree on ~95% of structures, diverge on thin pockets

Radii values in `rosetta_radii.py` are **numeric constants** from the published
`sc_radii.lib` parameter file (Lawrence & Colman contact radii). Reimplementing
those lookups from the data file is not a code port.

---

## 11. Tests

```bash
PYTHONPATH=src python -m pytest tests/ -q
```

`tests/test_trim_fallback.py` covers trim guard, target fallback, reverse matched
source.

---

## 12. Suggested next steps for a follow-up agent

1. Wire `save_scores.py` to use `score_msms_with_sphere_fallback` and write
   `used_sphere_fallback` column
2. Generate updated correlation plot from `sc_scores_hybrid.csv`
3. Optionally expose `surface_backend` flag in batch scoring for auditing
4. Long-term: implement a clean-room Connolly dot generator (paper-based) if MSMS
   + sphere hybrid ever proves insufficient — Rosetta source should remain unread

---

## 13. One-paragraph recap

Build MSMS solvent-excluded surfaces (protein atom pre-filter 8 Å), mark dots
whose probe overlaps the opponent, trim peripheral rim (with protein trim guard),
score bidirectional nearest-neighbor normal complementarity with several
sparse-interface fallbacks; **if MSMS protein trimmed dots < 2000**, rebuild
surfaces with a Fibonacci-sphere approximation and score with the same logic.
This hybrid achieves Pearson/Spearman ~0.97 vs Rosetta on 2,385 protein–ligand
poses while MSMS alone preserves ranking but fails on ~2% of structures.
