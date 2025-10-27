#!/usr/bin/env python
"""
Build HierarchicalLibrary for each dataset
This is the FIRST STEP before training
"""

from torch_geometric.datasets import MoleculeNet
from rdkit import Chem
from pathlib import Path
import json

# Corrected imports for flat directory
from substructure_library import SubstructureLibrary
from adaptive_library import HierarchicalLibrary

def prepare_data(dataset):
    """Add RDKit mol objects"""
    processed = []
    for data in dataset:
        try:
            if hasattr(data, 'smiles'):
                mol = Chem.MolFromSmiles(data.smiles)
                if mol is not None:
                    data.mol = mol
                    processed.append(data)
        except:
            continue
    return processed

def build_library(dataset_name, config):
    """Build and save hierarchical library"""
    print(f"\n{'='*70}")
    print(f"Building Library for: {dataset_name}")
    print(f"{'='*70}")
    
    # Load dataset
    print(f"\n[1/4] Loading {dataset_name} dataset...")
    try:
        dataset = MoleculeNet(root='data', name=dataset_name)
        print(f"  Loaded {len(dataset)} molecules")
    except Exception as e:
        print(f"  ✗ Failed to load: {e}")
        return None
    
    # Prepare
    print(f"\n[2/4] Processing molecules...")
    processed = prepare_data(dataset)
    print(f"  Valid molecules: {len(processed)}")
    
    if len(processed) < 10:
        print("  ✗ Too few molecules, skipping...")
        return None
    
    # Create base library
    print(f"\n[3/4] Creating HierarchicalLibrary...")
    base_library = SubstructureLibrary(include_extended=True)
    
    hierarchical_lib = HierarchicalLibrary(
        base_library=base_library,
        redundancy_threshold=config['redundancy_threshold'],
        min_frequency=config['min_frequency'],
        max_substructures_per_level=config.get('max_per_level')
    )
    
    # Fit to data
    hierarchical_lib.fit(processed)
    
    # Print statistics
    hierarchical_lib.get_statistics()
    
    # Save
    print(f"\n[4/4] Saving library...")
    Path('libraries').mkdir(exist_ok=True)
    filepath = f'libraries/{dataset_name.lower()}_hierarchical.json'
    hierarchical_lib.save(filepath)
    
    return hierarchical_lib

def main():
    print("="*70)
    print("HIERARCHICAL LIBRARY BUILDER")
    print("="*70)
    print("\nThis script builds dataset-specific hierarchical libraries.")
    print("Run this BEFORE training SA-GNN models.\n")
    
    # Dataset configurations
    datasets = {
        'ESOL': {
            'redundancy_threshold': 0.90,
            'min_frequency': 0.03,
            'max_per_level': {
                'atoms': 9,
                'bonds': 15,
                'small_rings': 8,
                'functional_groups': 15,
                'motifs': 10
            }
        },
        'FreeSolv': {
            'redundancy_threshold': 0.90,
            'min_frequency': 0.03,
            'max_per_level': {
                'atoms': 9,
                'bonds': 15,
                'small_rings': 10,
                'functional_groups': 20,
                'motifs': 10
            }
        },
        'HIV': {
            'redundancy_threshold': 0.90,
            'min_frequency': 0.03,
            'max_per_level': {
                'atoms': 9,
                'bonds': 15,
                'small_rings': 12,
                'functional_groups': 25,
                'motifs': 15
            }
        },
        'BACE': {
            'redundancy_threshold': 0.90,
            'min_frequency': 0.03,
            'max_per_level': {
                'atoms': 9,
                'bonds': 15,
                'small_rings': 10,
                'functional_groups': 20,
                'motifs': 12
            }
        },
        'Tox21': {
            'redundancy_threshold': 0.90,
            'min_frequency': 0.03,
            'max_per_level': {
                'atoms': 9,
                'bonds': 15,
                'small_rings': 10,
                'functional_groups': 20,
                'motifs': 12
            }
        }
    }
    
    # Build all libraries
    results = {}
    
    for dataset_name, config in datasets.items():
        try:
            lib = build_library(dataset_name, config)
            
            if lib:
                results[dataset_name] = {
                    'success': True,
                    'n_substructures': len(lib.pattern_list),
                    'base_size': len(lib.base_library.pattern_list),
                    'reduction_pct': (1 - len(lib.pattern_list) / len(lib.base_library.pattern_list)) * 100
                }
        except Exception as e:
            print(f"\n✗ Error with {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            results[dataset_name] = {'success': False, 'error': str(e)}
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Dataset':<15} {'Status':<10} {'Substructs':<12} {'Reduction':<12}")
    print("-"*70)
    
    for dataset_name, info in results.items():
        if info['success']:
            print(f"{dataset_name:<15} {'✓':<10} "
                  f"{info['n_substructures']:<12} "
                  f"{info['reduction_pct']:.1f}%")
        else:
            print(f"{dataset_name:<15} {'✗':<10} {'Failed':<12} {'-':<12}")
    
    print("="*70)
    
    # Save summary
    with open('libraries/build_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Summary saved to libraries/build_summary.json")
    print(f"✓ All libraries saved to libraries/ directory")
    print(f"\nYou can now train SA-GNN models with these libraries!")

if __name__ == '__main__':
    main()