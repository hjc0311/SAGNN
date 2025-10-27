"""
Substructure-aware pooling with guaranteed one-to-one mapping
"""

import torch
import torch.nn as nn
from typing import Dict, List, Set, Tuple

class SubstructureAwarePooling(nn.Module):
    """
    Pool node features with explicit substructure correspondence.
    Each substructure type gets dedicated representation dimensions.
    This enforces the one-to-one mapping property.
    """
    
    def __init__(self, 
                 node_dim: int, 
                 dim_per_substructure: int, 
                 substructure_names: List[str],
                 aggregation: str = 'sum'):
        """
        Args:
            node_dim: Dimension of node features from GNN
            dim_per_substructure: Dimension allocated to each substructure
            substructure_names: Ordered list of substructure names
            aggregation: How to aggregate multiple instances ('sum', 'mean', 'max')
        """
        super().__init__()
        
        self.node_dim = node_dim
        self.dim_per_substructure = dim_per_substructure
        self.n_substructures = len(substructure_names)
        self.substructure_names = substructure_names
        
        # Standardize aggregation
        if aggregation in ['sum', 'add']:
            self.aggregation = 'sum'
        elif aggregation in ['mean', 'avg']:
            self.aggregation = 'mean'
        else:
            print(f"Warning: Unknown aggregation '{aggregation}'. Defaulting to 'sum'.")
            self.aggregation = 'sum'
        
        # One encoder per substructure type (key for one-to-one mapping)
        self.substructure_encoders = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(node_dim, dim_per_substructure),
                nn.ReLU(),
                # nn.Dropout(0.1), # Dropout here can break zero-mapping
                nn.Linear(dim_per_substructure, dim_per_substructure)
            )
            for name in substructure_names
        })
    
    def forward(self, 
                node_features: torch.Tensor, 
                detections: Dict[str, List[Tuple[int, ...]]], # <-- CORRECTED HINT
                batch_idx: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            node_features: [n_nodes, node_dim] from GNN backbone
            detections: {substructure_name: [list_of_atom_index_tuples]}
            batch_idx: [n_nodes] batch assignment (for batched graphs)
            
        Returns:
            Representation tensor:
            - if batched: [batch_size, n_substructures * dim_per_substructure]
            - if single: [n_substructures * dim_per_substructure]
        """
        device = node_features.device
        
        # Determine batch size
        batch_size = 1
        if batch_idx is not None:
            batch_size = int(batch_idx.max().item()) + 1
        
        # Initialize output representation for all items in batch
        full_representation = torch.zeros(
            batch_size, 
            self.n_substructures * self.dim_per_substructure, 
            device=device
        )
        
        for i, sub_name in enumerate(self.substructure_names):
            start_dim = i * self.dim_per_substructure
            end_dim = start_dim + self.dim_per_substructure
            
            if sub_name in detections and len(detections[sub_name]) > 0:
                
                # --- START FIX ---
                # detections[sub_name] is a list of tuples, e.g., [(0, 1, 2), (4, 5, 6)]
                matches = detections[sub_name]
                
                # Flatten the list of tuples into a single list of atoms
                # e.g., [0, 1, 2, 4, 5, 6]
                atom_indices = [atom_idx for match in matches for atom_idx in match]
                
                if not atom_indices:
                    # Should not happen if check above passed, but good practice
                    continue
                
                # Convert to tensor
                indices = torch.tensor(atom_indices,
                                     dtype=torch.long, 
                                     device=device)
                
                # Get features of all atoms for this substructure
                sub_node_features = node_features[indices] # [n_atoms_in_batch, node_dim]
                
                # Get batch assignment for these specific atoms
                atom_batch_idx = batch_idx[indices] if batch_idx is not None else torch.zeros(indices.numel(), dtype=torch.long, device=device)
                
                # Apply substructure-specific encoder
                # [n_atoms_in_batch, dim_per_substructure]
                encoded_features = self.substructure_encoders[sub_name](sub_node_features)

                # Aggregate features per graph in the batch
                # This will sum features for each graph
                if self.aggregation == 'sum':
                    aggregated = torch.zeros(batch_size, self.dim_per_substructure, device=device)
                    aggregated.index_add_(0, atom_batch_idx, encoded_features)
                
                elif self.aggregation == 'mean':
                    aggregated = torch.zeros(batch_size, self.dim_per_substructure, device=device)
                    aggregated.index_add_(0, atom_batch_idx, encoded_features)
                    
                    # Compute counts for mean
                    counts = torch.bincount(atom_batch_idx, minlength=batch_size).float().clamp(min=1).unsqueeze(1)
                    aggregated = aggregated / counts
                
                # Place aggregated features into the correct slice
                full_representation[:, start_dim:end_dim] = aggregated
                # --- END FIX ---
            
            # else:
            #   No need for else, representation is already zeros
        
        # Handle single graph case (not batched)
        if batch_idx is None:
            return full_representation.squeeze(0) # [repr_dim]
        else:
            return full_representation # [batch_size, repr_dim]
    
    def get_substructure_slice(self, substructure_name: str) -> Tuple[int, int]:
        """
        Get slice indices for a specific substructure in the representation
        
        This is crucial for interpretability and ablation studies
        """
        if substructure_name not in self.substructure_names:
            raise ValueError(f"Substructure {substructure_name} not found")
        
        idx = self.substructure_names.index(substructure_name)
        start = idx * self.dim_per_substructure
        end = start + self.dim_per_substructure
        return start, end
    
    def get_substructure_representation(self, 
                                       full_representation: torch.Tensor,
                                       substructure_name: str) -> torch.Tensor:
        """Extract representation for a specific substructure"""
        start, end = self.get_substructure_slice(substructure_name)
        
        if full_representation.dim() == 2: # Batch
            return full_representation[:, start:end]
        else: # Single
            return full_representation[start:end]