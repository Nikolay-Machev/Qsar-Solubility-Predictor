"""SMILES validation and reproducible RDKit descriptor generation."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


def molecule_from_smiles(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    return molecule


def calculate_descriptors(smiles, descriptor_names=None):
    values = Descriptors.CalcMolDescriptors(
        molecule_from_smiles(smiles), missingVal=np.nan, silent=True
    )
    names = list(descriptor_names) if descriptor_names is not None else list(values)
    missing = [name for name in names if name not in values]
    if missing:
        raise ValueError(f"RDKit does not provide saved descriptors: {missing}")
    row = np.asarray([values[name] for name in names], dtype=float)
    row[~np.isfinite(row)] = np.nan
    return row, names


def scaffold_key(smiles):
    molecule = molecule_from_smiles(smiles)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
    # Acyclic molecules have an empty Bemis-Murcko scaffold; keep distinct
    # molecules separate instead of collapsing every acyclic compound together.
    return scaffold or f"acyclic:{Chem.MolToSmiles(molecule, canonical=True)}"
