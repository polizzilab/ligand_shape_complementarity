import numpy as np

from shape_complementarity import score_complex


def test_sc_atom_radius_routes_by_residue():
    from shape_complementarity.radii import sc_atom_radius

    assert sc_atom_radius("ALA", "CB", "C") == 1.95
    assert sc_atom_radius("LIG", "C1", "C") == sc_atom_radius("XXX", "C1", "C")
    assert sc_atom_radius("LIG", "O1", "O") < sc_atom_radius("ALA", "CB", "C")


def test_ligand_radii_from_smiles_apixaban(apixaban_pdb, apixaban_smiles):
    import prody as pr
    from shape_complementarity.ligand_radii import ligand_radii_from_smiles, map_ligand_from_smiles

    lig = pr.parsePDB(str(apixaban_pdb)).select("hetero and not water and not hydrogen")

    mapped = map_ligand_from_smiles(apixaban_smiles, lig)
    radii = ligand_radii_from_smiles(apixaban_smiles, lig)

    assert mapped.GetNumAtoms() == lig.numAtoms()
    assert radii.shape == (lig.numAtoms(),)
    assert 1.5 < float(radii.mean()) < 1.8


def test_score_complex_smoke():
    a_xyz = np.array([[0.0, 0.0, 0.0], [2.8, 0.0, 0.0]], dtype=np.float32)
    a_radii = np.array([1.7, 1.7], dtype=np.float32)
    b_xyz = np.array([[1.4, 0.0, 3.0]], dtype=np.float32)
    b_radii = np.array([1.7], dtype=np.float32)

    result = score_complex(a_xyz, a_radii, b_xyz, b_radii, n_sphere_points=32)

    assert result.n_surface_points_a > 0
    assert result.n_surface_points_b > 0
    assert result.n_trimmed_points_a >= 0
    assert result.n_trimmed_points_b >= 0
