"""Van der Waals radii used for surface construction."""

from __future__ import annotations

from rdkit import Chem

from .rosetta_radii import rosetta_sc_radius

_PERIODIC_TABLE = Chem.GetPeriodicTable()
_DEFAULT_ELEMENT = "C"

CANONICAL_AMINO_ACIDS = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }
)


def normalize_element(element: str) -> str:
    """Normalize an element symbol from a PDB parser."""
    element = (element or "").strip()
    if len(element) > 1:
        return element[0].upper() + element[1:].lower()
    return element.upper()


def radius_for_element(element: str) -> float:
    """Return an RDKit periodic-table van der Waals radius in Angstrom."""
    normalized = normalize_element(element)
    try:
        atomic_number = _PERIODIC_TABLE.GetAtomicNumber(normalized)
    except RuntimeError:
        atomic_number = 0
    if atomic_number <= 0:
        atomic_number = _PERIODIC_TABLE.GetAtomicNumber(_DEFAULT_ELEMENT)
    return radius_for_atomic_number(atomic_number)


def radius_for_atomic_number(atomic_number: int) -> float:
    """Return an RDKit periodic-table VdW radius for an atomic number."""
    if atomic_number <= 0:
        atomic_number = _PERIODIC_TABLE.GetAtomicNumber(_DEFAULT_ELEMENT)
    return float(_PERIODIC_TABLE.GetRvdw(atomic_number))


def sc_atom_radius(resname: str, atom_name: str, element: str) -> float:
    """Return the SC surface radius for one atom.

    Canonical amino-acid residues use Rosetta SC contact radii from
    ``sc_radii.lib``.  All other residues (ligands, cofactors, ions) use RDKit
    periodic-table VdW radii keyed by element.
    """
    if resname.strip().upper() in CANONICAL_AMINO_ACIDS:
        return rosetta_sc_radius(resname, atom_name)
    return radius_for_element(element)
