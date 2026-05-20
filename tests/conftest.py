import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTDATA = ROOT / "testdata"

sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def testdata_dir() -> Path:
    return TESTDATA


@pytest.fixture(scope="session")
def apixaban_smiles() -> str:
    return "COC1=CC=C(N2C3=C(C(C(N)=O)=N2)CCN(C4=CC=C(N5CCCCC5=O)C=C4)C3=O)C=C1"


APX_EXAMPLE_PDB = "pdbs/0001.pdb.gz"


@pytest.fixture(scope="session")
def apixaban_pdb(testdata_dir: Path) -> Path:
    return testdata_dir / APX_EXAMPLE_PDB
