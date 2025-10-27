import numpy as np
import json
from scipy.stats import pearsonr
from typing import Dict, List
from rdkit import Chem

# Imports from other files in the same directory
from substructure_library import SubstructureLibrary
from detector import SubstructureDetector


class HierarchicalLibrary:
    """
    Build library hierarchically: atoms → bonds → rings → groups → motifs
    Only include higher-level patterns if they add information
    """
    
    def __init__(self, 
                 base_library: SubstructureLibrary = None,
                 redundancy_threshold: float = 0.90,
                 min_frequency: float = 0.01,
                 max_substructures_per_level: Dict[str, int] = None):
        """
        Args:
            base_library: Full library of possible substructures
            redundancy_threshold: Correlation threshold for redundancy (0-1)
            min_frequency: Minimum frequency to include substructure
            max_substructures_per_level: Max substructures per hierarchy level
        """
        self.base_library = base_library or SubstructureLibrary(include_extended=True)
        self.redundancy_threshold = redundancy_threshold
        self.min_frequency = min_frequency
        
        # Default limits per level
        self.max_per_level = max_substructures_per_level or {
            'atoms': 9,
            'bonds': 15,
            'small_rings': 10,
            'functional_groups': 20,
            'motifs': 15
        }
        
        self.selected_patterns = {}
        self.pattern_list = []
        self.frequencies = {}
        self.level_info = {}  # Store which level each pattern belongs to
    
    def fit(self, dataset):
        """
        Build library hierarchically, removing redundant patterns
        
        Args:
            dataset: List of molecules or PyG dataset
        """
        print(f"\nBuilding hierarchical library from {len(dataset)} molecules...")
        print(f"Redundancy threshold: {self.redundancy_threshold}")
        print(f"Min frequency: {self.min_frequency}")
        
        detector = SubstructureDetector(self.base_library)
        
        # 1. Compute feature matrix for all substructures
        print("\n[1/3] Computing feature matrix...")
        features, valid_count = self._compute_feature_matrix(dataset, detector)
        
        if valid_count == 0:
            raise ValueError("No valid molecules found in dataset")
        
        # 2. Compute frequencies
        print("\n[2/3] Computing frequencies...")
        frequencies = {
            name: np.sum(vec) / valid_count
            for name, vec in features.items()
        }
        
        # 3. Select hierarchically
        print("\n[3/3] Hierarchical selection...")
        selected = []
        level_counts = {}
        
        hierarchy_levels = ['atoms', 'bonds', 'small_rings', 'functional_groups', 'motifs']
        
        for level_idx, level in enumerate(hierarchy_levels):
            print(f"\n  Level {level_idx + 1}/5: {level.upper()}")
            
            if level not in self.base_library.substructures:
                print(f"    Skipping (not in library)")
                continue
            
            level_patterns = list(self.base_library.substructures[level].keys())
            added_count = 0
            skipped_freq = 0
            skipped_redundant = 0
            
            # Sort by frequency (prioritize common patterns)
            level_patterns.sort(key=lambda x: frequencies.get(x, 0), reverse=True)
            
            for name in level_patterns:
                # Check frequency threshold
                if frequencies.get(name, 0) < self.min_frequency:
                    skipped_freq += 1
                    continue
                
                # Check if already at max for this level
                if added_count >= self.max_per_level.get(level, 999):
                    break
                
                # Check redundancy with existing patterns
                is_redundant, corr_with = self._check_redundancy(
                    name, selected, features
                )
                
                if is_redundant:
                    skipped_redundant += 1
                    print(f"    ✗ {name:25s}: redundant with {corr_with}")
                    continue
                
                # Add pattern
                selected.append(name)
                self.level_info[name] = level
                added_count += 1
                freq = frequencies[name]
                print(f"    ✓ {name:25s}: freq={freq:.3f}")
            
            level_counts[level] = added_count
            print(f"    Summary: {added_count} added, {skipped_freq} low-freq, {skipped_redundant} redundant")
        
        # 4. Create final library
        print(f"\n" + "="*70)
        print(f"✓ Hierarchical library built successfully!")
        print(f"  Total selected: {len(selected)} substructures")
        print(f"  Original library: {len(self.base_library.pattern_list)} substructures")
        print(f"  Reduction: {100*(1-len(selected)/len(self.base_library.pattern_list)):.1f}%")
        
        print(f"\nBreakdown by level:")
        for level, count in level_counts.items():
            print(f"  {level:20s}: {count:3d} substructures")
        
        all_patterns = self.base_library.get_all_patterns()
        self.selected_patterns = {
            name: all_patterns[name]
            for name in selected
        }
        
        self.pattern_list = selected
        self.frequencies = {name: frequencies[name] for name in selected}
        
        return self
    
    def _compute_feature_matrix(self, dataset, detector):
        """Compute binary feature matrix for all substructures"""
        features = {name: [] for name in self.base_library.pattern_list}
        valid_count = 0
        
        for i, data in enumerate(dataset):
            # if i % 100 == 0:
            #     print(f"    Processed {i}/{len(dataset)} molecules...")
            
            mol = self._get_mol(data)
            if mol is None:
                continue
            
            detections = detector.detect(mol)
            
            for name in self.base_library.pattern_list:
                present = 1.0 if name in detections else 0.0
                features[name].append(present)
            
            valid_count += 1
        
        # Convert to numpy arrays
        features = {
            name: np.array(vec) 
            for name, vec in features.items()
        }
        
        print(f"    ✓ Feature matrix computed: {valid_count} molecules")
        
        return features, valid_count
    
    def _check_redundancy(self, candidate_name, existing_patterns, features):
        """
        Check if candidate is redundant with existing patterns
        
        Returns:
            (is_redundant, correlated_with_pattern)
        """
        if not existing_patterns:
            return False, None
        
        candidate_vec = features[candidate_name]
        
        # Skip if candidate never appears
        if np.sum(candidate_vec) == 0:
            return True, "never_appears"
        
        for existing_name in existing_patterns:
            existing_vec = features[existing_name]
            
            # Skip if existing never appears
            if np.sum(existing_vec) == 0:
                continue
            
            # Compute correlation
            try:
                # Use Pearson correlation
                corr, _ = pearsonr(candidate_vec, existing_vec)
                
                # Also check Jaccard similarity for binary data
                intersection = np.sum(candidate_vec * existing_vec)
                union = np.sum(np.maximum(candidate_vec, existing_vec))
                jaccard = intersection / union if union > 0 else 0
                
                # Consider redundant if either metric is high
                if abs(corr) > self.redundancy_threshold or jaccard > self.redundancy_threshold:
                    return True, f"{existing_name} (corr={corr:.2f}, jaccard={jaccard:.2f})"
                
            except Exception as e:
                # If correlation fails, skip
                continue
        
        return False, None
    
    def _get_mol(self, data):
        """Extract RDKit mol from various data formats"""
        if hasattr(data, 'mol'):
            return data.mol
        elif hasattr(data, 'smiles'):
            from rdkit import Chem
            return Chem.MolFromSmiles(data.smiles)
        elif isinstance(data, str):
            from rdkit import Chem
            return Chem.MolFromSmiles(data)
        return None
    
    def get_all_patterns(self):
        """Return selected patterns"""
        return self.selected_patterns
    
    def get_statistics(self):
        """Print detailed statistics"""
        print("\n" + "="*70)
        print("Hierarchical Substructure Library")
        print("="*70)
        print(f"Total selected: {len(self.pattern_list)}")
        print(f"Redundancy threshold: {self.redundancy_threshold}")
        print(f"Min frequency: {self.min_frequency}")
        
        # Group by level
        for level in ['atoms', 'bonds', 'small_rings', 'functional_groups', 'motifs']:
            level_patterns = [
                name for name in self.pattern_list
                if self.level_info.get(name) == level
            ]
            
            if level_patterns:
                print(f"\n{level.upper().replace('_', ' ')} ({len(level_patterns)}):")
                for name in sorted(level_patterns, key=lambda x: self.frequencies[x], reverse=True):
                    freq = self.frequencies[name]
                    print(f"  {name:25s}: {freq:.3f}")
    
    def get_level_indices(self, level: str) -> List[int]:
        """
        Get indices of patterns at specific hierarchy level
        
        Args:
            level: 'atoms', 'bonds', 'small_rings', 'functional_groups', or 'motifs'
        
        Returns:
            List of indices in pattern_list
        """
        return [
            i for i, name in enumerate(self.pattern_list)
            if self.level_info.get(name) == level
        ]
    
    def visualize_hierarchy(self):
        """Create visualization of hierarchy structure"""
        import matplotlib.pyplot as plt
        import networkx as nx
        
        G = nx.DiGraph()
        
        # Add nodes by level
        pos = {}
        y_spacing = 1.0
        
        for level_idx, level in enumerate(['atoms', 'bonds', 'small_rings', 'functional_groups', 'motifs']):
            level_patterns = [
                name for name in self.pattern_list
                if self.level_info.get(name) == level
            ]
            
            x_spacing = 2.0 / (len(level_patterns) + 1)
            for i, name in enumerate(level_patterns):
                G.add_node(name, level=level)
                pos[name] = ((i + 1) * x_spacing, -level_idx * y_spacing)
        
        # Draw
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Color by level
        color_map = {
            'atoms': '#e74c3c',
            'bonds': '#e67e22',
            'small_rings': '#f39c12',
            'functional_groups': '#2ecc71',
            'motifs': '#3498db'
        }
        
        node_colors = [color_map[self.level_info[node]] for node in G.nodes()]
        
        nx.draw(G, pos, 
               node_color=node_colors,
               node_size=500,
               with_labels=True,
               font_size=6,
               font_weight='bold',
               ax=ax)
        
        ax.set_title('Hierarchical Substructure Library', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def save(self, filepath):
        """Save library configuration"""
        import json
        
        data = {
            'pattern_list': self.pattern_list,
            'frequencies': self.frequencies,
            'level_info': self.level_info,
            'redundancy_threshold': self.redundancy_threshold,
            'min_frequency': self.min_frequency,
            'max_per_level': self.max_per_level
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Library saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load saved library"""
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        library = cls(
            redundancy_threshold=data['redundancy_threshold'],
            min_frequency=data['min_frequency'],
            max_substructures_per_level=data['max_per_level']
        )
        
        library.pattern_list = data['pattern_list']
        library.frequencies = data['frequencies']
        library.level_info = data['level_info']
        
        all_patterns = library.base_library.get_all_patterns()
        library.selected_patterns = {
            name: all_patterns[name]
            for name in library.pattern_list
            if name in all_patterns
        }
        
        return library



class AdaptiveSubstructureLibrary:
    """
    Dataset-specific substructure library based on frequency analysis
    """
    
    def __init__(self, 
                 base_library: SubstructureLibrary,
                 min_frequency: float = 0.01,  # 1% of molecules
                 max_substructures: int = 100):
        """
        Args:
            base_library: Full library of possible substructures
            min_frequency: Minimum frequency to include substructure
            max_substructures: Maximum number of substructures to keep
        """
        self.base_library = base_library
        self.min_frequency = min_frequency
        self.max_substructures = max_substructures
        self.selected_patterns = {}
        self.pattern_list = []      # Initialize pattern_list
        self.frequencies = {}   # Initialize frequencies
    
    def fit(self, dataset):
        """
        Analyze dataset and select most relevant substructures
        
        Args:
            dataset: List of molecules or PyG dataset
        """
        from collections import Counter
        from rdkit import Chem
        
        print(f"Analyzing {len(dataset)} molecules...")
        
        # Count substructure frequencies
        substructure_counts = Counter()
        detector = SubstructureDetector(self.base_library)
        
        for i, data in enumerate(dataset):
            # if i % 100 == 0:
            #     print(f"  Processed {i}/{len(dataset)} molecules...")
            
            mol = data.mol if hasattr(data, 'mol') else Chem.MolFromSmiles(data.smiles)
            if mol is None:
                continue
            
            detections = detector.detect(mol)
            
            # Count which substructures appear
            for substructure_name in detections.keys():
                substructure_counts[substructure_name] += 1
        
        # Calculate frequencies
        n_molecules = len(dataset)
        substructure_freq = {
            name: count / n_molecules 
            for name, count in substructure_counts.items()
        }
        
        # Select substructures above threshold
        selected = {
            name: freq 
            for name, freq in substructure_freq.items()
            if freq >= self.min_frequency
        }
        
        # Sort by frequency and take top-k
        selected = dict(
            sorted(selected.items(), key=lambda x: x[1], reverse=True)
            [:self.max_substructures]
        )
        
        print(f"\n✓ Selected {len(selected)} substructures:")
        print(f"  Frequency range: {min(selected.values()):.3f} - {max(selected.values()):.3f}")
        
        # Get the pattern dictionary by *calling the method*
        all_base_patterns = self.base_library.get_all_patterns()

        # Create filtered patterns
        self.selected_patterns = {
            name: all_base_patterns[name]
            for name in selected.keys()
        }
        
        self.pattern_list = list(self.selected_patterns.keys())
        self.frequencies = selected
        
        return self

    # --- START FIX ---
    # This method was missing, causing the crash.
    def get_all_patterns(self):
        """Return selected patterns"""
        return self.selected_patterns
    # --- END FIX ---
    
    def get_statistics(self):
        """Print statistics about selected substructures"""
        print("\n" + "="*70)
        print("Selected Substructures")
        print("="*70)
        
        # Group by level
        # --- FIX IMPORT ---
        # from sagnn.data.substructure_library import SubstructureLibrary
        # This import was relative to a package, but we are in a flat directory
        base = SubstructureLibrary()
        # --- END FIX ---
        
        for level in ['atoms', 'bonds', 'small_rings', 'functional_groups', 'motifs']:
            level_patterns = [
                name for name in self.pattern_list
                if name in base.substructures[level]
            ]
            
            if level_patterns:
                print(f"\n{level.upper()} ({len(level_patterns)}):")
                for name in level_patterns[:10]:  # Top 10
                    print(f"  {name:25s}: {self.frequencies[name]:.3f}")
                if len(level_patterns) > 10:
                    print(f"  ... and {len(level_patterns)-10} more")
    
    def save(self, filepath):
        """Save selected library"""
        import json
        
        data = {
            'pattern_list': self.pattern_list,
            'frequencies': self.frequencies,
            'min_frequency': self.min_frequency,
            'max_substructures': self.max_substructures
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✓ Library saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load saved library"""
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        library = cls(
            base_library=SubstructureLibrary(),
            min_frequency=data['min_frequency'],
            max_substructures=data['max_substructures']
        )
        
        library.pattern_list = data['pattern_list']
        library.frequencies = data['frequencies']
        
        # Get the pattern dictionary by *calling the method*
        all_base_patterns = library.base_library.get_all_patterns()
        
        library.selected_patterns = {
            name: all_base_patterns[name]
            for name in library.pattern_list
        }
        
        return library