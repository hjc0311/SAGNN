"""
Complete SA-GNN model with support for custom libraries
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union

from torch_geometric.nn import global_mean_pool

# Corrected imports for flat directory
from substructure_library import SubstructureLibrary
from detector import SubstructureDetector
from backbone import SAGNNBackbone
from pooling import SubstructureAwarePooling


class SAGNN(nn.Module):
    """
    Substructure-Anchored Graph Neural Network
    """

    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 hidden_dim: int,
                 dim_per_substructure: int,
                 n_layers: int,
                 output_dim: int,
                 task: str = 'regression',
                 dropout: float = 0.1,
                 pooling_aggregation: str = 'sum',
                 substructure_library = None,
                 library_type: str = 'auto',
                 use_global_pool: bool = True):
        super().__init__()

        self.task = task
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.dim_per_substructure = dim_per_substructure
        self.output_dim = output_dim
        self.library_type = library_type
        self.use_global_pool = use_global_pool

        self._initialize_library(substructure_library, library_type)

        self.backbone = SAGNNBackbone(
            node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim,
            n_layers=n_layers, dropout=dropout
        )

        self.pooling = SubstructureAwarePooling(
            node_dim=hidden_dim, dim_per_substructure=dim_per_substructure,
            substructure_names=self.detector.pattern_list,
            aggregation=pooling_aggregation
        )

        self.representation_dim = len(self.detector.pattern_list) * dim_per_substructure
        self.final_repr_dim = self.representation_dim

        if self.use_global_pool:
            self.global_pool_projection = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
            )
            self.final_repr_dim += hidden_dim

        print(f"\n{'='*70}")
        print(f"SA-GNN Architecture Summary")
        print(f"{'='*70}")
        print(f"Library type: {self.library_type}")
        print(f"Number of substructures: {len(self.detector.pattern_list)}")
        print(f"Substructure Repr Dim: {self.representation_dim}")
        print(f"Use Global Pool: {self.use_global_pool}")
        print(f"Final Representation Dim: {self.final_repr_dim}")

        self.predictor = nn.Sequential(
            nn.Linear(self.final_repr_dim, self.final_repr_dim // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.final_repr_dim // 2, self.final_repr_dim // 4), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.final_repr_dim // 4, output_dim)
        )

        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:,}")
        print(f"{'='*70}\n")

        if task == 'classification':
            self.output_activation = nn.Sigmoid() if output_dim == 1 else nn.Softmax(dim=-1)


    def _initialize_library(self, substructure_library, library_type):
        """Initialize substructure library based on type"""
        if substructure_library is not None:
            print(f"Using custom {type(substructure_library).__name__}...")
            self.library = substructure_library
            if library_type == 'auto':
                lib_name = type(substructure_library).__name__
                if 'Hierarchical' in lib_name: self.library_type = 'hierarchical'
                elif 'Information' in lib_name or 'MI' in lib_name: self.library_type = 'mi'
                elif 'Adaptive' in lib_name: self.library_type = 'adaptive'
                else: self.library_type = 'custom'
        else:
            if library_type in ['auto', 'default']:
                print("Using default SubstructureLibrary...")
                self.library = SubstructureLibrary(include_extended=True)
                self.library_type = 'default'
            else:
                raise ValueError(f"library_type='{library_type}' requires substructure_library")

        self.detector = SubstructureDetector(self.library)
        print(f"Loaded {len(self.detector.pattern_list)} substructure patterns")


    def forward(self, data):
        """ Forward pass """
        node_features = self.backbone(data.x, data.edge_index, data.edge_attr)
        is_batch = hasattr(data, 'batch') and data.batch is not None

        # --- Substructure Detection ---
        detections = {}
        if is_batch:
            # Use mol_list if available, fallback to single mol
            mols = data.mol_list if hasattr(data, 'mol_list') else (data.mol if hasattr(data, 'mol') else [])
            node_offsets = data.ptr[:-1].cpu()  # Ensure offsets are CPU
            for i, mol in enumerate(mols):
                if mol is None: continue  # Skip if RDKit failed for this mol
                mol_detections = self.detector.detect(mol)
                node_offset = node_offsets[i].item()
                for sub_name, matches in mol_detections.items():
                    offset_matches = [tuple(idx + node_offset for idx in m) for m in matches]
                    detections.setdefault(sub_name, []).extend(offset_matches)
        else:  # Single data object
            if hasattr(data, 'mol') and data.mol is not None:
                detections = self.detector.detect(data.mol)
            # else: detections remains empty if no mol object

        # Rest of the forward pass remains unchanged
        substructure_representation = self.pooling(
            node_features,
            detections,
            batch_idx=data.batch if is_batch else None
        )

        if self.use_global_pool:
            batch_vec = data.batch if is_batch else torch.zeros(
                node_features.size(0), dtype=torch.long, device=node_features.device
            )
            global_representation = global_mean_pool(node_features, batch_vec)
            global_representation_proj = self.global_pool_projection(global_representation)

            if not is_batch:
                if substructure_representation.dim() == 1:
                    substructure_representation = substructure_representation.unsqueeze(0)
                if global_representation_proj.dim() == 1:
                    global_representation_proj = global_representation_proj.unsqueeze(0)

            if substructure_representation.dim() != global_representation_proj.dim():
                raise RuntimeError(f"Dimension mismatch before concat: Substructure {substructure_representation.shape}, Global {global_representation_proj.shape}")

            representation = torch.cat([substructure_representation, global_representation_proj], dim=-1)
        else:
            representation = substructure_representation

        prediction = self.predictor(representation)

        return prediction, representation, detections


    def get_substructure_importance(self, data, task_name: str = None, method: str = 'ablation') -> Dict[str, float]:
        """ Compute importance of each substructure (for single-task model) """
        # task_name is ignored for single-task, but kept for API consistency
        
        self.eval()
        model_device = next(self.parameters()).device
        data = data.to(model_device)

        with torch.no_grad():
            # base_pred is the raw score
            base_pred, base_repr, detections = self.forward(data) 
            base_pred_value = base_pred.squeeze().item()

        importances = {}
        for sub_name in self.detector.pattern_list:
            ablated_repr = base_repr.clone()
            start, end = self.pooling.get_substructure_slice(sub_name)

            if ablated_repr.dim() == 2:
                ablated_repr[:, start:end] = 0
            else:
                ablated_repr[start:end] = 0

            with torch.no_grad():
                ablated_input = ablated_repr if ablated_repr.dim() == 2 else ablated_repr.unsqueeze(0)
                # Use the main predictor
                ablated_pred = self.predictor(ablated_input)
                ablated_pred_value = ablated_pred.squeeze().item()

            importance = abs(base_pred_value - ablated_pred_value)
            importances[sub_name] = importance
        return importances


    def get_hierarchical_importance(self, data) -> Dict[str, Dict[str, float]]:
        """
        Get importance organized by hierarchy level (only for HierarchicalLibrary)
        """
        if self.library_type != 'hierarchical':
            raise ValueError("This method only works with HierarchicalLibrary")

        importances = self.get_substructure_importance(data)
        hierarchical_imp = {}

        for level in ['atoms', 'bonds', 'small_rings', 'functional_groups', 'motifs']:
            level_patterns = [
                name for name in self.detector.pattern_list
                if hasattr(self.library, 'level_info') and self.library.level_info.get(name) == level
            ]

            if level_patterns:
                hierarchical_imp[level] = {
                    name: importances[name]
                    for name in level_patterns
                }

        return hierarchical_imp

    def print_hierarchical_importance(self, data, top_k: int = 5):
        """
        Print importance scores organized by hierarchy
        """
        if self.library_type != 'hierarchical':
            print("Warning: Not using HierarchicalLibrary")
            return

        hierarchical_imp = self.get_hierarchical_importance(data)

        print("\n" + "="*70)
        print("Hierarchical Substructure Importance")
        print("="*70)

        for level in ['atoms', 'bonds', 'small_rings', 'functional_groups', 'motifs']:
            if level not in hierarchical_imp:
                continue

            level_imp = hierarchical_imp[level]

            print(f"\n{level.upper().replace('_', ' ')}:")

            sorted_imp = sorted(level_imp.items(), key=lambda x: x[1], reverse=True)

            for i, (name, score) in enumerate(sorted_imp[:top_k], 1):
                print(f"  {i}. {name:25s}: {score:.4f}")

            if len(sorted_imp) > top_k:
                print(f"  ... and {len(sorted_imp) - top_k} more")

    def verify_one_to_one_mapping(self, data) -> Tuple[bool, Dict[str, bool]]:
        """ Verify the one-to-one mapping property """
        # Ensure data is on the correct device
        model_device = next(self.parameters()).device
        data = data.to(model_device)
        with torch.no_grad():
            _, representation, detections = self.forward(data)

        # Verification needs CPU representation and original mol object
        cpu_representation = representation.cpu()
        cpu_mol = data.cpu().mol # Get mol from CPU version of data

        if cpu_mol is None:
             print("Warning: Cannot verify mapping, mol object not found.")
             return False, {}

        results = self.detector.verify_one_to_one_mapping(
            cpu_representation, cpu_mol, self.dim_per_substructure
        )

        all_valid = all(results.values())
        return all_valid, results

    def get_library_info(self) -> Dict:
        """Get information about the substructure library"""
        info = { 'type': self.library_type,
                 'n_substructures': len(self.detector.pattern_list),
                 'representation_dim': self.representation_dim,
                 'dim_per_substructure': self.dim_per_substructure }
        if self.library_type == 'hierarchical' and hasattr(self.library, 'level_info'):
            info['hierarchy_levels'] = {}
            for level in ['atoms', 'bonds', 'small_rings', 'functional_groups', 'motifs']:
                count = sum(1 for name in self.detector.pattern_list if self.library.level_info.get(name) == level)
                info['hierarchy_levels'][level] = count
        if hasattr(self.library, 'frequencies'):
             info['avg_frequency'] = (sum(self.library.frequencies.values()) / len(self.library.frequencies)) if self.library.frequencies else 0.0
        return info


class SAGNNMultiTask(SAGNN):
    """
    Multi-task variant of SA-GNN for simultaneous property prediction
    Also supports all library types
    """
    
    def __init__(self, 
                 node_dim: int,
                 edge_dim: int,
                 hidden_dim: int,
                 dim_per_substructure: int,
                 n_layers: int,
                 task_output_dims: Dict[str, int],
                 task_types: Dict[str, str],
                 substructure_library = None,
                 library_type: str = 'auto',
                 **kwargs):
        """
        Args:
            task_output_dims: Dictionary mapping task_name -> output_dim
            task_types: Dictionary mapping task_name -> 'regression' or 'classification'
            substructure_library: Custom library (any type)
            library_type: Library type hint
        """
        # Initialize with dummy output_dim
        super().__init__(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            dim_per_substructure=dim_per_substructure,
            n_layers=n_layers,
            output_dim=1,
            substructure_library=substructure_library,
            library_type=library_type,
            **kwargs
        )
        
        self.task_names = list(task_output_dims.keys())
        self.task_types = task_types
        
        # Replace single predictor with task-specific heads
        self.task_predictors = nn.ModuleDict({
            task_name: nn.Sequential(
                nn.Linear(self.representation_dim, self.representation_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.representation_dim // 2, task_output_dims[task_name])
            )
            for task_name in self.task_names
        })
        
        # Task-specific output activations
        self.task_activations = {}
        for task_name, task_type in task_types.items():
            if task_type == 'classification':
                if task_output_dims[task_name] == 1:
                    self.task_activations[task_name] = nn.Sigmoid()
                else:
                    self.task_activations[task_name] = nn.Softmax(dim=-1)
    
    def forward(self, data):
        """Forward pass for all tasks"""
        # Get representation
        node_features = self.backbone(data.x, data.edge_index, data.edge_attr)
        detections = self.detector.detect(data.mol)
        representation = self.pooling(node_features, detections)
        
        # Predict all tasks
        predictions = {}
        for task_name in self.task_names:
            pred = self.task_predictors[task_name](representation)
            
            if task_name in self.task_activations:
                pred = self.task_activations[task_name](pred)
            
            predictions[task_name] = pred
        
        return predictions, representation, detections

    def get_substructure_importance(self, data, task_name: str = None, method: str = 'ablation') -> Dict[str, float]:
        """ Compute importance of each substructure for a specific task """
        if task_name is None:
            task_name = list(self.task_predictors.keys())[0]

        if task_name not in self.task_predictors:
            raise ValueError(f"Task {task_name} not found in model")

        self.eval()
        model_device = next(self.parameters()).device
        data = data.to(model_device)

        with torch.no_grad():
            base_pred, base_repr, detections = self.forward(data)
            base_pred_value = base_pred[task_name].squeeze().item()

        importances = {}
        for sub_name in self.detector.pattern_list:
            ablated_repr = base_repr.clone()
            start, end = self.pooling.get_substructure_slice(sub_name)

            if ablated_repr.dim() == 2:
                ablated_repr[:, start:end] = 0
            else:
                ablated_repr[start:end] = 0

            with torch.no_grad():
                ablated_input = ablated_repr if ablated_repr.dim() == 2 else ablated_repr.unsqueeze(0)
                ablated_pred = self.task_predictors[task_name](ablated_input)
                ablated_pred_value = ablated_pred.squeeze().item()

            importance = abs(base_pred_value - ablated_pred_value)
            importances[sub_name] = importance
        return importances