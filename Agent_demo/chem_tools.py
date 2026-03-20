from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from pathlib import Path
import os

base_dir = Path("./test")

def validate_smiles(smiles: str) -> str:
    """Checks if a SMILES string is valid.
    
    Args:
        smiles: The SMILES string to check.
        
    Returns:
        A message indicating validity.
    """
    print(f"(validate_smiles {smiles})")
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return f"Valid SMILES: {smiles}"
    else:
        return f"Invalid SMILES: {smiles}"

def get_molecule_descriptors(smiles: str) -> str:
    """Calculates basic molecular descriptors (MW, LogP).
    
    Args:
        smiles: The SMILES string of the molecule.
        
    Returns:
        A string containing molecular weight and LogP, or error message.
    """
    print(f"(get_molecule_descriptors {smiles})")
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return f"Invalid SMILES: {smiles}"
    
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    
    return f"Molecular Weight: {mw:.2f}\nLogP: {logp:.2f}"

def find_substructure(target_smiles: str, query_smarts: str) -> str:
    """Checks if a substructure exists in the target molecule.
    
    Args:
        target_smiles: The SMILES string of the target molecule.
        query_smarts: The SMARTS string of the substructure to find.
        
    Returns:
        A message indicating if a match was found.
    """
    print(f"(find_substructure {target_smiles} {query_smarts})")
    target = Chem.MolFromSmiles(target_smiles)
    if not target:
        return f"Invalid Target SMILES: {target_smiles}"
        
    query = Chem.MolFromSmarts(query_smarts)
    if not query:
        return f"Invalid Query SMARTS: {query_smarts}"
        
    if target.HasSubstructMatch(query):
        return f"Substructure found in {target_smiles}"
    else:
        return f"Substructure NOT found in {target_smiles}"

def generate_2d_image(smiles: str, filename: str) -> str:
    """Generates a 2D image of the molecule and saves it to the test directory.
    
    Args:
        smiles: The SMILES string.
        filename: The filename to save (e.g., 'mol.png').
    
    Returns:
        Status message.
    """
    print(f"(generate_2d_image {smiles} -> {filename})")
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return f"Invalid SMILES: {smiles}"
        
    try:
        # Security check similar to rename_file
        save_path = base_dir / filename
        if ".." in str(save_path) or not str(save_path.resolve()).startswith(str(base_dir.resolve())):
             return "Error: Invalid filename or path traversal detected."

        # Add explicit hydrogens to make the image look better (especially for small molecules like Water)
        mol_with_hs = Chem.AddHs(mol)
        Draw.MolToFile(mol_with_hs, str(save_path))
        return f"Image saved to {filename}"
    except Exception as e:
        return f"Error generating image: {e}"
