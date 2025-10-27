"""
Hierarchical library of molecular substructures
"""

from rdkit import Chem
from typing import Dict, List, Tuple

class SubstructureLibrary:
    """
    Comprehensive library of molecular substructures organized hierarchically
    """
    
    def __init__(self, include_extended: bool = True):
        """
        Args:
            include_extended: Include extended substructure set (Level 5+)
        """
        self.include_extended = include_extended
        
        # Level 1: Atoms (9 types)
        self.atoms = {
            'C': '[C]',
            'N': '[N]',
            'O': '[O]',
            'F': '[F]',
            'Cl': '[Cl]',
            'Br': '[Br]',
            'S': '[S]',
            'P': '[P]',
            'I': '[I]'
        }
        
        # Level 2: Bonds (14 types)
        self.bonds = {
            'C-C': '[C]-[C]',
            'C=C': '[C]=[C]',
            'C#C': '[C]#[C]',
            'C-N': '[C]-[N]',
            'C=N': '[C]=[N]',
            'C-O': '[C]-[O]',
            'C=O': '[C]=[O]',
            'C-S': '[C]-[S]',
            'C=S': '[C]=[S]',
            'N-N': '[N]-[N]',
            'N=N': '[N]=[N]',
            'N-O': '[N]-[O]',
            'O-O': '[O]-[O]',
            'S-S': '[S]-[S]'
        }
        
        # Level 3: Small Rings (8 types)
        self.small_rings = {
            'benzene': 'c1ccccc1',
            'pyrrole': 'c1cc[nH]c1',
            'pyridine': 'c1ccncc1',
            'furan': 'c1ccoc1',
            'thiophene': 'c1ccsc1',
            'imidazole': 'c1c[nH]cn1',
            'cyclohexane': 'C1CCCCC1',
            'cyclopentane': 'C1CCCC1'
        }
        
        # Level 4: Functional Groups (15 types)
        self.functional_groups = {
            'carboxyl': 'C(=O)O',
            'amine_primary': '[NX3;H2]',
            'amine_secondary': '[NX3;H1]',
            'amine_tertiary': '[NX3;H0]',
            'hydroxyl': '[OX2H]',
            'carbonyl': '[CX3]=[OX1]',
            'ester': 'C(=O)OC',
            'amide': 'C(=O)N',
            'nitro': 'N(=O)=O',
            'nitrile': 'C#N',
            'sulfone': 'S(=O)(=O)',
            'sulfonamide': 'S(=O)(=O)N',
            'phosphate': 'P(=O)(O)(O)',
            'aldehyde': '[CX3H1](=O)',
            'ketone': '[CX3](=O)[#6]'
        }
        
        # Level 5: Complex Motifs (10 types)
        if include_extended:
            self.motifs = {
                'peptide_bond': 'NC(=O)',
                'urea': 'NC(=O)N',
                'guanidine': 'NC(=N)N',
                'sulfonamide': 'S(=O)(=O)N',
                'carbamate': 'NC(=O)O',
                'thiourea': 'NC(=S)N',
                'hydrazone': 'C=NN',
                'imine': 'C=N',
                'enamine': 'C=CN',
                'acetal': 'C(O)O'
            }
        else:
            self.motifs = {}
        
        # Compile all substructures
        self.substructures = {
            'atoms': self.atoms,
            'bonds': self.bonds,
            'small_rings': self.small_rings,
            'functional_groups': self.functional_groups,
            'motifs': self.motifs
        }
        
        # Compile SMARTS patterns
        self.patterns = {}
        self.pattern_list = []
        
        for level, substructs in self.substructures.items():
            self.patterns[level] = {}
            for name, smarts in substructs.items():
                try:
                    pattern = Chem.MolFromSmarts(smarts)
                    if pattern is not None:
                        self.patterns[level][name] = pattern
                        self.pattern_list.append(name)
                    else:
                        print(f"Warning: Could not parse SMARTS for {name}: {smarts}")
                except Exception as e:
                    print(f"Error parsing {name}: {e}")
        
        print(f"Loaded {len(self.pattern_list)} substructure patterns")
    
    def get_all_patterns(self) -> Dict[str, Chem.Mol]:
        """Flatten hierarchy into single dict"""
        all_patterns = {}
        for level_patterns in self.patterns.values():
            all_patterns.update(level_patterns)
        return all_patterns
    
    def get_pattern_index(self, pattern_name: str) -> int:
        """Get index of pattern in ordered list"""
        return self.pattern_list.index(pattern_name)
    
    def get_level_indices(self, level: str) -> List[int]:
        """Get indices of all patterns at a given level"""
        if level not in self.patterns:
            return []
        return [self.get_pattern_index(name) for name in self.patterns[level].keys()]