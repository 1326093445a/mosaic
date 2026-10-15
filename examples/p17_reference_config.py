"""Default reference identity, importable without loading a model."""

from pathlib import Path

COMPLEX_PDB = Path(__file__).resolve().parent.parent / "P17_JN1.pdb"
BINDER_CHAIN = "B"
TARGET_CHAIN = "T"
