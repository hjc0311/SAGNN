"""
Substructure detection using graph isomorphism
"""

import torch
from rdkit import Chem
from typing import Dict, List, Set, Tuple
import numpy as np

class SubstructureDetector:
    """
    Detects and localizes substructures in molecular graphs
    Uses RDKit's subgraph isomorphism (VF2 algorithm)
    """
    
    def __init__(self, library):
        """
        Args:
            library: SubstructureLibrary instance
        """
        self.library = library
        self.all_patterns = library.get_all_patterns()
        self.pattern_list = library.pattern_list
        self.n_patterns = len(self.pattern_list)
        
        # Cache for detections
        self.cache = {}
        self.cache_enabled = True
    
    def enable_cache(self, enabled: bool = True):
        """Enable/disable detection caching"""
        self.cache_enabled = enabled
        if not enabled:
            self.cache.clear()
    
    def detect(self, mol: Chem.Mol, use_cache: bool = True) -> Dict[str, List[Tuple[int, ...]]]:
        """
        Detect all substructures in molecule
        
        Args:
            mol: RDKit molecule object
            use_cache: Use cached results if available
            
        Returns:
            Dictionary mapping substructure_name -> list of atom index tuples
        """
        if mol is None:
            return {}
        
        # Check cache
        mol_hash = Chem.MolToSmiles(mol)
        if use_cache and self.cache_enabled and mol_hash in self.cache:
            return self.cache[mol_hash]
        
        detections = {}
        
        for pattern_name in self.pattern_list:
            pattern = self.all_patterns[pattern_name]
            if pattern is None:
                continue
            
            try:
                matches = mol.GetSubstructMatches(pattern)
                if matches:
                    detections[pattern_name] = matches
            except Exception as e:
                # Skip patterns that cause errors
                continue
        
        # Cache results
        if use_cache and self.cache_enabled:
            self.cache[mol_hash] = detections
        
        return detections
    
    def detect_batch(self, mols: List[Chem.Mol]) -> List[Dict[str, List[Tuple[int, ...]]]]:
        """
        Detect substructures in batch of molecules
        Can be parallelized for speedup
        """
        return [self.detect(mol) for mol in mols]
    
    def create_detection_vector(self, mol: Chem.Mol, 
                               count_instances: bool = True) -> np.ndarray:
        """
        Create vector indicating presence/count of each substructure
        
        Args:
            mol: RDKit molecule
            count_instances: If True, count instances; if False, binary presence
            
        Returns:
            Array of shape [n_substructures]
        """
        detections = self.detect(mol)
        vector = np.zeros(self.n_patterns)
        
        for idx, pattern_name in enumerate(self.pattern_list):
            if pattern_name in detections:
                if count_instances:
                    vector[idx] = len(detections[pattern_name])
                else:
                    vector[idx] = 1.0
        
        return vector
    
    def create_detection_tensor(self, mol: Chem.Mol, 
                               device: str = 'cpu') -> torch.Tensor:
        """Create PyTorch tensor of detection vector"""
        vector = self.create_detection_vector(mol, count_instances=True)
        return torch.tensor(vector, dtype=torch.float32, device=device)
    
    def get_atom_to_substructures(self, mol: Chem.Mol) -> Dict[int, List[str]]:
        """
        Map each atom to list of substructures it belongs to
        
        Returns:
            Dictionary mapping atom_idx -> [substructure_names]
        """
        detections = self.detect(mol)
        atom_map = {i: [] for i in range(mol.GetNumAtoms())}
        
        for substructure_name, matches in detections.items():
            for match in matches:
                for atom_idx in match:
                    atom_map[atom_idx].append(substructure_name)
        
        return atom_map
    
    def verify_one_to_one_mapping(self, representation: torch.Tensor, 
                                  mol: Chem.Mol,
                                  dim_per_substructure: int) -> Dict[str, bool]:
        """
        Verify that representation follows one-to-one mapping property
        
        Returns:
            Dictionary mapping substructure_name -> is_valid
        """
        detections = self.detect(mol)
        results = {}
        
        for idx, pattern_name in enumerate(self.pattern_list):
            start = idx * dim_per_substructure
            end = start + dim_per_substructure
            repr_slice = representation[start:end]
            
            # Check: present in molecule ⟺ non-zero representation
            is_present = pattern_name in detections and len(detections[pattern_name]) > 0
            is_nonzero = torch.norm(repr_slice) > 1e-6
            
            results[pattern_name] = (is_present == is_nonzero)
        
        return results