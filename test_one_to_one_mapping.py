"""
Test to verify one-to-one mapping property
"""

import torch
from rdkit import Chem

from sagnn import SAGNN
from substructure_library import SubstructureLibrary
from detector import SubstructureDetector
from torch_geometric.data import Data



def create_test_molecule(smiles):
    """Create test data from SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    
    atom_features = []
    for atom in mol.GetAtoms():
        atom_type = [0] * 9
        try:
            atom_idx = ['C', 'N', 'O', 'F', 'Cl', 'Br', 'S', 'P', 'I'].index(atom.GetSymbol())
        except:
            atom_idx = 0
        atom_type[atom_idx] = 1
        atom_features.append(atom_type)
    
    x = torch.tensor(atom_features, dtype=torch.float)
    
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]])
        
        # --- START FIX ---
        # Explicitly create the 4-dim one-hot encoding
        bond_type = [0] * 4  # [SINGLE, DOUBLE, TRIPLE, AROMATIC]
        bt = bond.GetBondType()
        
        if bt == Chem.BondType.SINGLE:
            bond_type[0] = 1
        elif bt == Chem.BondType.DOUBLE:
            bond_type[1] = 1
        elif bt == Chem.BondType.TRIPLE:
            bond_type[2] = 1
        elif bt == Chem.BondType.AROMATIC:
            bond_type[3] = 1
        # --- END FIX ---
        
        edge_attr.extend([bond_type, bond_type])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.mol = mol
    
    return data

def test_one_to_one_mapping():
    """Test that representation follows one-to-one mapping"""
    
    model = SAGNN(
        node_dim=9,
        edge_dim=4,
        hidden_dim=64,
        dim_per_substructure=16,
        n_layers=2,
        output_dim=1
    )
    
    model.eval()
    
    # Test molecules
    test_smiles = [
        'CC(C)(C(=O)N1CCOc2c(C#N)cncc2C1)C(F)F',  # DNL-788
        'CN1C[C@H]2C[C@@]2(C#Cc2cc3ncnc(Nc4cccc(Cl)c4F)c3cc2NC(=O)/C=C/CN2CCOCC2)C1',   # BDTX-1535
        'Cc1cc(Nc2ncc3c(n2)N(C)CN(c2c(Cl)cccc2Cl)C3=O)ccc1C1CCN(C)CC1',       # Debio-0123e
        'CN1CCc2cc(Nc3ncc(C4CC4)c(NCCCNC(=O)C4CCC4)n3)ccc2C1', # MRT-68921
        'CO[C@H]1CCN(c2nccc(Nc3cc4c(C(C)C)ccc(N5CC(CS(C)(=O)=O)C5)c4cn3)n2)C[C@H]1F'   # Blueprint-63
    ]
    
    all_valid = True
    
    for smiles in test_smiles:
        data = create_test_molecule(smiles)
        
        with torch.no_grad():
            is_valid, results = model.verify_one_to_one_mapping(data)
        
        print(f"\n{smiles}: {'✓ VALID' if is_valid else '✗ INVALID'}")
        
        if not is_valid:
            all_valid = False
            print("  Failed substructures:")
            for name, valid in results.items():
                if not valid:
                    print(f"    - {name}")
    
    assert all_valid, "One-to-one mapping property violated!"
    print("\n✓ All tests passed!")

def test_substructure_detection():
    """Test substructure detection"""
    
    library = SubstructureLibrary()
    detector = SubstructureDetector(library)
    
    # Test benzene detection
    benzene = Chem.MolFromSmiles('c1ccccc1')
    detections = detector.detect(benzene)
    
    assert 'benzene' in detections, "Benzene not detected in benzene molecule!"
    print("✓ Benzene detection: PASS")
    
    # Test carboxyl detection
    acetic_acid = Chem.MolFromSmiles('CC(=O)O')
    detections = detector.detect(acetic_acid)
    
    assert 'carboxyl' in detections, "Carboxyl not detected in acetic acid!"
    print("✓ Carboxyl detection: PASS")
    
    # Test amine detection
    ethylamine = Chem.MolFromSmiles('CCN')
    detections = detector.detect(ethylamine)
    
    assert any('amine' in key for key in detections.keys()), "Amine not detected in ethylamine!"
    print("✓ Amine detection: PASS")

def test_ablation_consistency():
    """Test that ablation changes predictions as expected"""
    
    model = SAGNN(
        node_dim=9,
        edge_dim=4,
        hidden_dim=64,
        dim_per_substructure=16,
        n_layers=2,
        output_dim=1
    )
    
    model.eval()
    
    # Test with Tucatinib
    data = create_test_molecule('Cc1cc(Nc2ncnc3ccc(NC4=NC(C)(C)CO4)cc23)ccc1Oc1ccn2ncnc2c1')
    
    with torch.no_grad():
        # Get baseline prediction
        pred1, repr1, detections = model(data)
        
        # Get importances
        importances = model.get_substructure_importance(data)
        
        # Ablate most important substructure
        most_important = max(importances.items(), key=lambda x: x[1])[0]
        
        repr_ablated = repr1.clone()
        start, end = model.pooling.get_substructure_slice(most_important)
        repr_ablated[start:end] = 0
        
        # Get ablated prediction
        pred2 = model.predictor(repr_ablated.unsqueeze(0))
    
    # Predictions should be different
    pred_diff = abs(pred1.item() - pred2.item())
    
    print(f"\nAblation test:")
    print(f"  Baseline prediction: {pred1.item():.4f}")
    print(f"  Ablated prediction: {pred2.item():.4f}")
    print(f"  Difference: {pred_diff:.4f}")
    
    assert pred_diff > 0.001, "Ablation did not change prediction!"
    print("✓ Ablation consistency: PASS")

if __name__ == '__main__':
    print("="*70)
    print("SA-GNN One-to-One Mapping Tests")
    print("="*70)
    
    print("\n1. Testing one-to-one mapping property...")
    test_one_to_one_mapping()
    
    print("\n2. Testing substructure detection...")
    test_substructure_detection()
    
    print("\n3. Testing ablation consistency...")
    test_ablation_consistency()
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED!")
    print("="*70)