"""ProDy-backed atom selection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import prody as pr

from .ligand_radii import ligand_radii_from_smiles
from .radii import sc_atom_radius


@dataclass(frozen=True)
class AtomSelection:
    """Coordinates and radii for a selected molecular component."""

    xyz: np.ndarray
    radii: np.ndarray
    elements: list[str]
    names: list[str]
    resnames: list[str]
    chains: list[str]


def _selection_to_atoms(selection, *, ligand_smiles: str | None = None) -> AtomSelection:
    if selection is None or selection.numAtoms() == 0:
        raise ValueError("ProDy selection matched zero atoms")

    elements = [str(element).strip() for element in selection.getElements()]
    names = [str(name).strip() for name in selection.getNames()]
    resnames = [str(resname).strip() for resname in selection.getResnames()]
    chains = [str(chain).strip() for chain in selection.getChids()]

    if ligand_smiles is not None:
        radii = ligand_radii_from_smiles(ligand_smiles, selection)
    else:
        radii = np.asarray(
            [sc_atom_radius(res, nm, elem) for res, nm, elem in zip(resnames, names, elements)],
            dtype=np.float32,
        )

    return AtomSelection(
        xyz=np.asarray(selection.getCoords(), dtype=np.float32),
        radii=radii,
        elements=elements,
        names=names,
        resnames=resnames,
        chains=chains,
    )


def _protein_radii(selection) -> np.ndarray:
    elements = [str(element).strip() for element in selection.getElements()]
    names = [str(name).strip() for name in selection.getNames()]
    resnames = [str(resname).strip() for resname in selection.getResnames()]
    return np.asarray(
        [sc_atom_radius(res, nm, elem) for res, nm, elem in zip(resnames, names, elements)],
        dtype=np.float32,
    )


def load_selection(
    path: str | Path,
    selection: str,
    *,
    ligand_smiles: str | None = None,
) -> AtomSelection:
    """Load atoms matching a ProDy selection string from a PDB file.

    Hydrogens are removed automatically.  Protein atoms use Rosetta SC contact
    radii for canonical amino acids.  When ``ligand_smiles`` is supplied, ligand
    radii come from an RDKit molecule mapped with ``AssignBondOrdersFromTemplate``.
    """
    atoms = pr.parsePDB(str(path))
    if atoms is None:
        raise ValueError(f"Could not parse PDB: {path}")
    selected = atoms.select(f"({selection}) and not hydrogen")
    return _selection_to_atoms(selected, ligand_smiles=ligand_smiles)


def load_protein_ligand(
    path: str | Path,
    *,
    protein_selection: str = "protein",
    ligand_selection: str = "hetero and not water",
    ligand_smiles: str | None = None,
) -> tuple[AtomSelection, AtomSelection]:
    """Load protein and ligand selections from one PDB.

    Pass ``ligand_smiles`` to map the ligand with RDKit and assign VdW radii from
    the mapped molecule rather than PDB element/name heuristics.
    """
    atoms = pr.parsePDB(str(path))
    if atoms is None:
        raise ValueError(f"Could not parse PDB: {path}")

    protein = atoms.select(f"({protein_selection}) and not hydrogen")
    ligand = atoms.select(f"({ligand_selection}) and not hydrogen")
    if protein is None or protein.numAtoms() == 0:
        raise ValueError(f"Protein selection matched zero atoms: {protein_selection!r}")
    if ligand is None or ligand.numAtoms() == 0:
        raise ValueError(f"Ligand selection matched zero atoms: {ligand_selection!r}")

    protein_atoms = AtomSelection(
        xyz=np.asarray(protein.getCoords(), dtype=np.float32),
        radii=_protein_radii(protein),
        elements=[str(element).strip() for element in protein.getElements()],
        names=[str(name).strip() for name in protein.getNames()],
        resnames=[str(resname).strip() for resname in protein.getResnames()],
        chains=[str(chain).strip() for chain in protein.getChids()],
    )
    ligand_atoms = _selection_to_atoms(ligand, ligand_smiles=ligand_smiles)
    return protein_atoms, ligand_atoms
