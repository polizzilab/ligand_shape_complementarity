"""Ligand radii from an RDKit molecule mapped onto a PDB ligand conformer."""

from __future__ import annotations

import io
from contextlib import contextmanager

import numpy as np
import prody as pr
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from .radii import radius_for_atomic_number

_COORD_TOL = 0.5


@contextmanager
def _suppress_rdkit_warnings():
    """Silence benign RDKit warnings during template mapping."""
    RDLogger.DisableLog("rdApp.warning")
    try:
        yield
    finally:
        RDLogger.EnableLog("rdApp.warning")


def map_ligand_from_smiles(smiles: str, ligand_selection) -> Chem.Mol:
    """Assign SMILES bond orders/charges onto a ProDy ligand selection.

    Mirrors the NISE pipeline pattern:
    ``AllChem.AssignBondOrdersFromTemplate(Chem.MolFromSmiles(smiles), pdb_mol)``.
    """
    if ligand_selection is None or ligand_selection.numAtoms() == 0:
        raise ValueError("Ligand selection matched zero atoms")

    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise ValueError(f"Invalid ligand SMILES: {smiles!r}")

    stream = io.StringIO()
    pr.writePDBStream(stream, ligand_selection)
    pdb_mol = Chem.MolFromPDBBlock(stream.getvalue(), removeHs=False)
    if pdb_mol is None:
        raise ValueError("RDKit could not parse the ligand PDB block")

    with _suppress_rdkit_warnings():
        try:
            mapped = AllChem.AssignBondOrdersFromTemplate(template, pdb_mol)
        except ValueError as exc:
            raise ValueError(
                "RDKit could not map ligand SMILES onto the PDB ligand geometry"
            ) from exc
    if mapped is None:
        raise ValueError("AssignBondOrdersFromTemplate returned no molecule")
    return mapped


def radii_from_mapped_ligand(mapped_mol: Chem.Mol) -> np.ndarray:
    """Return RDKit periodic-table VdW radii for each atom in ``mapped_mol``."""
    return np.asarray(
        [radius_for_atomic_number(atom.GetAtomicNum()) for atom in mapped_mol.GetAtoms()],
        dtype=np.float32,
    )


def _match_atoms_by_coordinates(
    ligand_xyz: np.ndarray,
    mapped_mol: Chem.Mol,
) -> np.ndarray:
    """Map ProDy ligand atom order to RDKit atom indices by 3D coordinates."""
    if mapped_mol.GetNumConformers() == 0:
        raise ValueError("Mapped ligand molecule has no conformer")

    conf = mapped_mol.GetConformer()
    rdkit_xyz = np.array(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
         for i in range(mapped_mol.GetNumAtoms())],
        dtype=np.float64,
    )

    radii = np.empty(len(ligand_xyz), dtype=np.float32)
    used: set[int] = set()
    for i, xyz in enumerate(ligand_xyz):
        dists = np.linalg.norm(rdkit_xyz - xyz, axis=1)
        order = np.argsort(dists)
        match_idx = None
        for idx in order:
            idx = int(idx)
            if idx in used:
                continue
            if dists[idx] <= _COORD_TOL:
                match_idx = idx
                break
        if match_idx is None:
            raise ValueError(
                f"Could not match ProDy ligand atom {i} to mapped RDKit coordinates "
                f"(closest distance {float(dists[order[0]]):.3f} Å)"
            )
        used.add(match_idx)
        radii[i] = radius_for_atomic_number(mapped_mol.GetAtomWithIdx(match_idx).GetAtomicNum())
    return radii


def ligand_radii_from_smiles(smiles: str, ligand_selection) -> np.ndarray:
    """Map ``smiles`` onto ``ligand_selection`` and return aligned VdW radii."""
    mapped = map_ligand_from_smiles(smiles, ligand_selection)
    ligand_xyz = np.asarray(ligand_selection.getCoords(), dtype=np.float64)

    if mapped.GetNumAtoms() == ligand_selection.numAtoms():
        direct = radii_from_mapped_ligand(mapped)
        if mapped.GetNumConformers() > 0:
            conf = mapped.GetConformer()
            mapped_xyz = np.array(
                [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                 for i in range(mapped.GetNumAtoms())],
                dtype=np.float64,
            )
            if np.max(np.linalg.norm(mapped_xyz - ligand_xyz, axis=1)) <= _COORD_TOL:
                return direct
        return _match_atoms_by_coordinates(ligand_xyz, mapped)

    return _match_atoms_by_coordinates(ligand_xyz, mapped)
