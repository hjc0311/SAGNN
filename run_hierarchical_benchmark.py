"""
Benchmark SA-GNN with HierarchicalLibrary
Complete comparison across all library types and datasets
"""

import torch
import torch.nn as nn
import json
import time
from pathlib import Path

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch.utils.data import random_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score
from scipy.stats import pearsonr
import numpy as np
from rdkit import Chem

from torch_geometric.data import Batch
import matplotlib.pyplot as plt
from rdkit.Chem import Draw

# Corrected imports for flat directory
from sagnn import SAGNN, SAGNNMultiTask
# --- CHANGE 1: Added Graphormer, GAT, MPNN to import ---
from baselines import GCN, GAT, MPNN, MorganFPModel, GIN, Graphormer
from substructure_library import SubstructureLibrary
from adaptive_library import HierarchicalLibrary, AdaptiveSubstructureLibrary

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")



def visualize_importances(model, data, metric_name, dataset_name, top_k=10, task_name=None):
    """
    Generates and saves a visualization of substructure importances
    """
    if not isinstance(model, SAGNN):
        print("  Visualization skipped: Not an SA-GNN model.")
        return

    print(f"    Generating importance visualization...")
    try:
        data_cpu = data.clone().to('cpu')
        model_device = next(model.parameters()).device
        if isinstance(model, SAGNNMultiTask):
            task_name = task_name or 'task_0'  # Default to first task
            importances = model.get_substructure_importance(data_cpu, task_name=task_name)
        else:
            importances = model.get_substructure_importance(data_cpu)

        if not hasattr(data_cpu, 'mol') or data_cpu.mol is None:
            print("    Skipping visualization: Mol object missing from sample data.")
            return
        img = Draw.MolToImage(data_cpu.mol, size=(300, 300))

        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        top_names = [name for name, score in sorted_imp[:top_k]]
        top_scores = [score for name, score in sorted_imp[:top_k]]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6),
                                       gridspec_kw={'width_ratios': [1, 2]})

        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title(f"SMILES: {Chem.MolToSmiles(data_cpu.mol)}")

        colors = plt.cm.viridis(np.linspace(0.4, 0.9, top_k))
        ax2.barh(np.arange(top_k), top_scores, color=colors, align='center')
        ax2.set_yticks(np.arange(top_k))
        ax2.set_yticklabels(top_names, fontsize=10)
        ax2.invert_yaxis()
        ax2.set_xlabel(f"Importance (Absolute Change in {metric_name})", fontsize=12)
        ax2.set_title(f"Top-10 Most Important Substructures ({task_name or 'default'})", fontsize=14)

        plt.tight_layout()

        save_dir = Path('results')
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{dataset_name.lower()}_{model.library_type}_importance_viz_{task_name or 'default'}.png"
        plt.savefig(save_path)
        plt.close(fig)
        print(f"    Visualization saved to {save_path}")

    except Exception as e:
        print(f"    Error during visualization: {e}")
        import traceback
        traceback.print_exc()

class HierarchicalBenchmark:
    """
    Extended benchmark for HierarchicalLibrary comparison
    """

    def __init__(self,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 results_dir: str = 'results'):
        self.device = device
        self.results_dir = Path(results_dir)

        # Ensure results directory exists robustly
        self.results_dir.mkdir(parents=True, exist_ok=True)

        print(f"Using device: {device}")

        self.results = {
            'library_comparison': {},
            'baseline_comparison': {},
            'efficiency_analysis': {},
            'interpretability': {}
        }


    def prepare_data(self, dataset):
        """Add RDKit mol objects to dataset and ensure all tensors live on CPU"""
        processed = []
        for i, data in enumerate(dataset):
            try:
                # Clone the data object to prevent in-place modifications
                data_clone = data.clone()

                # Ensure all tensor-like attributes are on CPU
                for attr in dir(data_clone):
                    if attr.startswith('_'):  # Skip private attributes
                        continue
                    val = getattr(data_clone, attr)
                    if isinstance(val, torch.Tensor):
                        setattr(data_clone, attr, val.cpu())
                    elif isinstance(val, list) and any(isinstance(item, torch.Tensor) for item in val):
                        # Handle lists of tensors if any
                        setattr(data_clone, attr, [item.cpu() if isinstance(item, torch.Tensor) else item for item in val])

                mol = None
                if hasattr(data_clone, 'smiles'):
                    mol = Chem.MolFromSmiles(data_clone.smiles)
                elif hasattr(data_clone, 'mol') and isinstance(data_clone.mol, Chem.Mol):
                    mol = data_clone.mol

                if mol is not None:
                    data_clone.mol = mol  # Keep RDKit mol on CPU
                    processed.append(data_clone)
            except Exception as e:
                print(f"Error processing data index {i}: {e}")
                continue
        return processed



    def create_model(self, model_type: str, dataset_info: dict, substructure_library=None):
        """ Create model instance (Moves model to self.device) """
        node_dim = dataset_info['node_dim']
        edge_dim = dataset_info.get('edge_dim', 4)
        output_dim = dataset_info['output_dim']
        task = dataset_info['task']
        is_multi_task = dataset_info.get('is_multi_task', False)
        task_output_dims = dataset_info.get('task_output_dims', None)
        task_types = dataset_info.get('task_types', None)

        if model_type.startswith('SAGNN'):
            if model_type == 'SAGNN_default': lib_type = 'default'; lib = None
            elif model_type == 'SAGNN_adaptive': lib_type = 'adaptive'; lib = substructure_library
            elif model_type == 'SAGNN_hierarchical': lib_type = 'hierarchical'; lib = substructure_library
            else: lib_type = 'auto'; lib = substructure_library

            if is_multi_task:
                model = SAGNNMultiTask(
                    node_dim=node_dim, edge_dim=edge_dim, hidden_dim=128,
                    dim_per_substructure=32, n_layers=3,
                    task_output_dims=task_output_dims, task_types=task_types,
                    dropout=0.2, substructure_library=lib, library_type=lib_type,
                    use_global_pool=True
                )
            else:
                model = SAGNN(
                    node_dim=node_dim, edge_dim=edge_dim, hidden_dim=128,
                    dim_per_substructure=32, n_layers=3, output_dim=output_dim, task=task,
                    dropout=0.2, substructure_library=lib, library_type=lib_type,
                    use_global_pool=True
                )
        elif model_type == 'GCN':
            model = GCN(node_dim=node_dim, hidden_dim=128, output_dim=output_dim, n_layers=3, dropout=0.2)
        elif model_type == 'GAT':
            model = GAT(node_dim=node_dim, hidden_dim=64, output_dim=output_dim, n_layers=3, heads=4, dropout=0.2)
        elif model_type == 'MPNN':
            model = MPNN(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=128, output_dim=output_dim, n_layers=3, dropout=0.2)
        elif model_type == 'MorganFP':
            model = MorganFPModel(fp_size=2048, radius=3, hidden_dim=512, output_dim=output_dim, dropout=0.3)
        elif model_type == 'GIN':
            model = GIN(node_dim=node_dim, hidden_dim=128, output_dim=output_dim, n_layers=3, dropout=0.2)
        # --- CHANGE 3: Added Graphormer instantiation logic ---
        elif model_type == 'Graphormer':
            model = Graphormer(
                node_dim=node_dim, 
                edge_dim=edge_dim, 
                hidden_dim=128,
                output_dim=output_dim,
                n_layers=3,
                n_heads=8,
                dropout=0.2
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return model.to(self.device)

    def train_model(self, model, train_loader, val_loader, task: str, model_name: str, is_multi_task: bool = False, epochs: int = 100, patience: int = 20):
        """Train a model with early stopping"""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        temp_model_path = self.results_dir / f'best_model_{model_name}.pt'

        if temp_model_path.exists():
            try:
                temp_model_path.unlink()
            except OSError as e:
                print(f"Warning: Could not delete stale model file {temp_model_path}: {e}")

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

        if is_multi_task:
            # Initialize loss functions for each task
            criterion = {
                task_name: nn.MSELoss() if task_type == 'regression' else nn.BCEWithLogitsLoss()
                for task_name, task_type in model.task_types.items()
            }
        else:
            criterion = nn.MSELoss() if task == 'regression' else nn.BCEWithLogitsLoss()

        best_val_loss = float('inf')
        patience_counter = 0
        training_time = 0

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            start_time = time.time()

            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()

                try:
                    if isinstance(model, SAGNNMultiTask):
                        pred, _, _ = model(batch)
                        loss = 0
                        for task_name in pred:
                            y = batch.y[:, int(task_name.split('_')[-1])].float()  # Select task-specific target
                            
                            # --- Tox21 FIX FOR MULTI-TASK (IF NEEDED) ---
                            is_valid = ~torch.isnan(y)
                            if not is_valid.all():
                                y_valid = y[is_valid]
                                pred_valid = pred[task_name].squeeze(-1)[is_valid]
                            else:
                                y_valid = y
                                pred_valid = pred[task_name].squeeze(-1)
                            
                            task_loss = criterion[task_name](pred_valid, y_valid) if y_valid.numel() > 0 else torch.tensor(0.0, device=self.device)
                            # --- END FIX ---
                            
                            loss += task_loss
                        loss = loss / len(pred)  # Average loss across tasks
                    
                    elif isinstance(model, SAGNN):
                        pred, _, _ = model(batch)
                        pred = pred.squeeze(-1) if pred.dim() > 1 and pred.size(-1) == 1 else pred
                        
                        y = batch.y
                        if y.dim() > 1 and y.size(-1) > 1:
                            y = y[:, 0] # Select first task
                            
                        y = y.squeeze(-1) if y.dim() > 1 and y.size(-1) == 1 else y
                        y = y.float()

                        # --- Tox21 FIX ---
                        is_valid = ~torch.isnan(y)
                        if not is_valid.all():
                            pred = pred[is_valid]
                            y = y[is_valid]
                        # --- END FIX ---
                        
                        loss = criterion(pred, y) if y.numel() > 0 else torch.tensor(0.0, device=self.device)

                    elif isinstance(model, MorganFPModel):
                        mols = [data.mol for data in batch.to_data_list() if hasattr(data, 'mol')]
                        batch.mol_list = mols
                        pred = model(batch)
                        pred = pred.squeeze(-1) if pred.dim() > 1 and pred.size(-1) == 1 else pred

                        y = batch.y
                        if y.dim() > 1 and y.size(-1) > 1:
                            y = y[:, 0] # Select first task

                        y = y.squeeze(-1) if y.dim() > 1 and y.size(-1) == 1 else y
                        y = y.float()
                        
                        # --- Tox21 FIX ---
                        is_valid = ~torch.isnan(y)
                        if not is_valid.all():
                            pred = pred[is_valid]
                            y = y[is_valid]
                        # --- END FIX ---

                        loss = criterion(pred, y) if y.numel() > 0 else torch.tensor(0.0, device=self.device)
                    
                    else: # GCN, GIN, GAT, MPNN, Graphormer
                        pred = model(batch)
                        pred = pred.squeeze(-1) if pred.dim() > 1 and pred.size(-1) == 1 else pred
                        
                        y = batch.y
                        if y.dim() > 1 and y.size(-1) > 1:
                            y = y[:, 0]  # Select first task
                        
                        y = y.squeeze(-1) if y.dim() > 1 and y.size(-1) == 1 else y
                        y = y.float()

                        # --- Tox21 FIX ---
                        is_valid = ~torch.isnan(y)
                        if not is_valid.all():
                            pred = pred[is_valid]
                            y = y[is_valid]
                        # --- END FIX ---

                        loss = criterion(pred, y) if y.numel() > 0 else torch.tensor(0.0, device=self.device)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    train_loss += loss.item()

                except Exception as e:
                    print(f"      Error in training batch: {e}")
                    continue

            training_time += time.time() - start_time
            train_loss /= (len(train_loader) if len(train_loader) > 0 else 1)

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    try:
                        if isinstance(model, SAGNNMultiTask):
                            pred, _, _ = model(batch)
                            task_loss_sum = 0
                            for task_name in pred:
                                y = batch.y[:, int(task_name.split('_')[-1])].float()

                                # --- Tox21 FIX FOR MULTI-TASK (IF NEEDED) ---
                                is_valid = ~torch.isnan(y)
                                if not is_valid.all():
                                    y_valid = y[is_valid]
                                    pred_valid = pred[task_name].squeeze(-1)[is_valid]
                                else:
                                    y_valid = y
                                    pred_valid = pred[task_name].squeeze(-1)
                                
                                if y_valid.numel() > 0:
                                    task_loss = criterion[task_name](pred_valid, y_valid)
                                    task_loss_sum += task_loss.item()
                                # --- END FIX ---

                            val_loss += task_loss_sum / len(pred)
                        
                        elif isinstance(model, SAGNN):
                            pred, _, _ = model(batch)
                            pred = pred.squeeze(-1) if pred.dim() > 1 and pred.size(-1) == 1 else pred
                            
                            y = batch.y
                            if y.dim() > 1 and y.size(-1) > 1:
                                y = y[:, 0]  # Select first task
                            
                            y = y.squeeze(-1) if y.dim() > 1 and y.size(-1) == 1 else y
                            y = y.float()

                            # --- Tox21 FIX ---
                            is_valid = ~torch.isnan(y)
                            if not is_valid.all():
                                pred = pred[is_valid]
                                y = y[is_valid]
                            # --- END FIX ---

                            if y.numel() > 0:
                                val_loss += criterion(pred, y).item()
                        
                        elif isinstance(model, MorganFPModel):
                            mols = [data.mol for data in batch.to_data_list() if hasattr(data, 'mol')]
                            batch.mol_list = mols
                            pred = model(batch)
                            pred = pred.squeeze(-1) if pred.dim() > 1 and pred.size(-1) == 1 else pred
                            
                            y = batch.y
                            if y.dim() > 1 and y.size(-1) > 1:
                                y = y[:, 0]  # Select first task
                            
                            y = y.squeeze(-1) if y.dim() > 1 and y.size(-1) == 1 else y
                            y = y.float()

                            # --- Tox21 FIX ---
                            is_valid = ~torch.isnan(y)
                            if not is_valid.all():
                                pred = pred[is_valid]
                                y = y[is_valid]
                            # --- END FIX ---

                            if y.numel() > 0:
                                val_loss += criterion(pred, y).item()
                        
                        else: # GCN, GIN, GAT, MPNN, Graphormer
                            pred = model(batch)
                            pred = pred.squeeze(-1) if pred.dim() > 1 and pred.size(-1) == 1 else pred
                            
                            y = batch.y
                            if y.dim() > 1 and y.size(-1) > 1:
                                y = y[:, 0]  # Select first task
                            
                            y = y.squeeze(-1) if y.dim() > 1 and y.size(-1) == 1 else y
                            y = y.float()

                            # --- Tox21 FIX ---
                            is_valid = ~torch.isnan(y)
                            if not is_valid.all():
                                pred = pred[is_valid]
                                y = y[is_valid]
                            # --- END FIX ---
                            
                            if y.numel() > 0:
                                val_loss += criterion(pred, y).item()

                    except Exception as e:
                        print(f"      Error in validation batch: {e}")
                        continue

                val_loss /= (len(val_loader) if len(val_loader) > 0 else 1)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), temp_model_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"    Early stopping at epoch {epoch+1}")
                        break

                if (epoch + 1) % 10 == 0:
                    print(f"    Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")

        if temp_model_path.exists():
            model.load_state_dict(torch.load(temp_model_path))
        else:
            print("Warning: No best model saved, possibly due to training errors.")

        return model, training_time

    def evaluate_regression(self, model, test_loader, is_multi_task: bool = False):
        """Evaluate regression metrics"""
        model.eval()
        predictions, targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                try:
                    if isinstance(model, SAGNNMultiTask):
                        pred, _, _ = model(batch)
                        # Collect predictions and targets for each task
                        task_preds = [pred[task_name].squeeze(-1).cpu().numpy() for task_name in pred]
                        task_targets = [batch.y[:, i].cpu().numpy() for i in range(batch.y.size(-1))]
                        predictions.append(task_preds)
                        targets.append(task_targets)
                    elif isinstance(model, SAGNN):
                        pred, _, _ = model(batch)
                        pred = pred.squeeze(-1) if pred.dim() > 1 and pred.size(-1) == 1 else pred
                        
                        y = batch.y
                        if y.dim() > 1 and y.size(-1) > 1:
                            y = y[:, 0]  # Select first task
                        
                        y = y.squeeze(-1) if y.dim() > 1 and y.size(-1) == 1 else y
                        predictions.extend(pred.cpu().numpy())
                        targets.extend(y.cpu().numpy())
                    elif isinstance(model, MorganFPModel):
                        mols = [data.mol for data in batch.to_data_list() if hasattr(data, 'mol')]
                        batch.mol_list = mols
                        pred = model(batch)
                        pred = pred.squeeze(-1) if pred.dim() > 1 and pred.size(-1) == 1 else pred
                        
                        y = batch.y
                        if y.dim() > 1 and y.size(-1) > 1:
                            y = y[:, 0]  # Select first task
                        
                        y = y.squeeze(-1) if y.dim() > 1 and y.size(-1) == 1 else y
                        predictions.extend(pred.cpu().numpy())
                        targets.extend(y.cpu().numpy())
                    else:
                        pred = model(batch)
                        pred = pred.squeeze(-1) if pred.dim() > 1 and pred.size(-1) == 1 else pred
                        
                        y = batch.y
                        if y.dim() > 1 and y.size(-1) > 1:
                            y = y[:, 0]  # Select first task
                        
                        y = y.squeeze(-1) if y.dim() > 1 and y.size(-1) == 1 else y
                        predictions.extend(pred.cpu().numpy())
                        targets.extend(y.cpu().numpy())
                except Exception as e:
                    print(f"      Error during regression evaluation batch: {e}")
                    continue

        if is_multi_task:
            # Compute metrics for each task
            metrics = {}
            num_tasks = batch.y.size(-1) # Get num_tasks from last batch
            predictions = np.array(predictions).transpose(1, 0, 2).reshape(num_tasks, -1)  # Shape: [num_tasks, n_samples]
            targets = np.array(targets).transpose(1, 0, 2).reshape(num_tasks, -1)
            for i in range(num_tasks):
                valid = ~(np.isnan(predictions[i]) | np.isnan(targets[i]))
                if np.any(valid):
                    mae = mean_absolute_error(targets[i][valid], predictions[i][valid])
                    rmse = np.sqrt(mean_squared_error(targets[i][valid], predictions[i][valid]))
                    r2 = pearsonr(targets[i][valid], predictions[i][valid])[0]**2
                    metrics[f'task_{i}'] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
                else:
                    metrics[f'task_{i}'] = {'MAE': float('nan'), 'RMSE': float('nan'), 'R2': float('nan')}
            # Average metrics across tasks
            metrics['MAE'] = np.mean([m['MAE'] for m in metrics.values() if not np.isnan(m['MAE'])])
            metrics['RMSE'] = np.mean([m['RMSE'] for m in metrics.values() if not np.isnan(m['RMSE'])])
            metrics['R2'] = np.mean([m['R2'] for m in metrics.values() if not np.isnan(m['R2'])])
            return metrics
        else:
            predictions, targets = np.array(predictions), np.array(targets)
            valid = ~(np.isnan(predictions) | np.isnan(targets))
            if not np.any(valid):
                return {'MAE': float('nan'), 'RMSE': float('nan'), 'R2': float('nan')}
            mae = mean_absolute_error(targets[valid], predictions[valid])
            rmse = np.sqrt(mean_squared_error(targets[valid], predictions[valid]))
            r2 = pearsonr(targets[valid], predictions[valid])[0]**2
            return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

    def evaluate_classification(self, model, test_loader, is_multi_task: bool = False):
        """Evaluate classification metrics"""
        model.eval()
        predictions, targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                try:
                    if isinstance(model, SAGNNMultiTask):
                        pred, _, _ = model(batch)
                        task_preds = [pred[task_name].squeeze(-1).cpu().numpy() for task_name in pred]
                        task_targets = [batch.y[:, i].cpu().numpy() for i in range(batch.y.size(-1))]
                        predictions.append(task_preds)
                        targets.append(task_targets)
                    elif isinstance(model, SAGNN):
                        pred, _, _ = model(batch)
                        pred = pred.squeeze(-1) if pred.dim() > 1 and pred.size(-1) == 1 else pred
                        
                        y = batch.y
                        if y.dim() > 1 and y.size(-1) > 1:
                            y = y[:, 0]  # Select first task
                        
                        y = y.squeeze(-1) if y.dim() > 1 and y.size(-1) == 1 else y
                        predictions.extend(pred.cpu().numpy())
                        targets.extend(y.cpu().numpy())
                    elif isinstance(model, MorganFPModel):
                        mols = [data.mol for data in batch.to_data_list() if hasattr(data, 'mol')]
                        batch.mol_list = mols
                        pred = model(batch)
                        pred = pred.squeeze(-1) if pred.dim() > 1 and pred.size(-1) == 1 else pred
                        
                        y = batch.y
                        if y.dim() > 1 and y.size(-1) > 1:
                            y = y[:, 0]  # Select first task
                        
                        y = y.squeeze(-1) if y.dim() > 1 and y.size(-1) == 1 else y
                        predictions.extend(pred.cpu().numpy())
                        targets.extend(y.cpu().numpy())
                    else:
                        pred = model(batch)
                        pred = pred.squeeze(-1) if pred.dim() > 1 and pred.size(-1) == 1 else pred
                        
                        y = batch.y
                        if y.dim() > 1 and y.size(-1) > 1:
                            y = y[:, 0]  # Select first task
                        
                        y = y.squeeze(-1) if y.dim() > 1 and y.size(-1) == 1 else y
                        predictions.extend(pred.cpu().numpy())
                        targets.extend(y.cpu().numpy())
                except Exception as e:
                    print(f"      Error during classification evaluation batch: {e}")
                    continue

        if is_multi_task:
            metrics = {}
            num_tasks = batch.y.size(-1) # Get num_tasks from last batch
            predictions = np.array(predictions).transpose(1, 0, 2).reshape(num_tasks, -1)
            targets = np.array(targets).transpose(1, 0, 2).reshape(num_tasks, -1)
            for i in range(num_tasks):
                valid = ~(np.isnan(predictions[i]) | np.isnan(targets[i]))
                if np.any(valid) and np.ptp(targets[i][valid]) > 0:
                    auc = roc_auc_score(targets[i][valid], predictions[i][valid])
                    metrics[f'task_{i}'] = {'ROC-AUC': auc}
                else:
                    metrics[f'task_{i}'] = {'ROC-AUC': float('nan')}
            metrics['ROC-AUC'] = np.mean([m['ROC-AUC'] for m in metrics.values() if not np.isnan(m['ROC-AUC'])])
            return metrics
        else:
            predictions, targets = np.array(predictions), np.array(targets)
            valid = ~(np.isnan(predictions) | np.isnan(targets))
            if not np.any(valid) or np.ptp(targets[valid]) == 0:
                return {'ROC-AUC': float('nan')}
            auc = roc_auc_score(targets[valid], predictions[valid])
            return {'ROC-AUC': auc}


    def run_hierarchical_comparison(self):
        """
        Main benchmark: Compare SA-GNN with different library types
        """
        print("\n" + "="*70)
        print("HIERARCHICAL LIBRARY BENCHMARK")
        print("="*70)

        datasets_to_test = ['ESOL', 'FreeSolv', 'BACE', 'HIV', 'Tox21']

        for dataset_name in datasets_to_test:
            print(f"\n{'='*70}")
            print(f"Dataset: {dataset_name}")
            print('='*70)

            try:
                # Load dataset (will be on CPU initially)
                dataset = MoleculeNet(root='data', name=dataset_name)
            except Exception as e:
                print(f"Error loading {dataset_name}: {e}")
                continue

            print("  Preparing data (ensuring CPU tensors and adding Mol objects)...")
            # prepare_data now ensures tensors are CPU and adds Mol objects
            processed_dataset_cpu = self.prepare_data(dataset)
            print(f"  Valid molecules: {len(processed_dataset_cpu)}")

            if len(processed_dataset_cpu) < 10:
                print(f"  Skipping {dataset_name}: not enough valid molecules")
                continue

            # Determine task type
            if dataset_name in ['BACE', 'HIV', 'Tox21']:
                task = 'classification'; metric_type = 'classification'; metric_name = 'ROC-AUC'
            else:
                task = 'regression'; metric_type = 'regression'; metric_name = 'RMSE'

            # Get dimensions from the first valid sample (on CPU)
            sample = processed_dataset_cpu[0]
            node_dim = sample.x.size(1)
            edge_dim = sample.edge_attr.size(1) if hasattr(sample, 'edge_attr') and sample.edge_attr is not None and sample.edge_attr.numel() > 0 else 0

            output_dim = 1 # Assuming single-task prediction

            dataset_info = {
                'node_dim': node_dim, 'edge_dim': edge_dim,
                'output_dim': output_dim, 'task': task
            }
            
            # If edge_dim is 0, MPNN needs adjustment or skipping
            if edge_dim == 0:
                 print("Warning: Edge features have dimension 0. MPNN might fail or needs adjustment.")


            # Split dataset (still CPU data)
            train_size = int(0.8 * len(processed_dataset_cpu))
            val_size = int(0.1 * len(processed_dataset_cpu))
            test_size = len(processed_dataset_cpu) - train_size - val_size

            # Ensure splits are valid
            if train_size <= 0 or val_size <= 0 or test_size <= 0:
                 print(f"  Skipping {dataset_name}: Invalid split sizes ({train_size}, {val_size}, {test_size})")
                 continue

            try:
                train_dataset, val_dataset, test_dataset = random_split(
                    processed_dataset_cpu, [train_size, val_size, test_size]
                )
            except ValueError as e:
                 print(f"  Skipping {dataset_name}: Error during split - {e}")
                 continue

            def custom_collate_fn(data_list):
                """
                Custom collate function to handle RDKit Mol objects and ensure CPU tensors.
                """
                # Filter out non-tensor attributes for PyTorch Geometric's Batch
                tensor_data_list = []
                mol_list = []
                for data in data_list:
                    tensor_data = data.__class__()  # Create new instance of same class
                    for attr in data.__dict__.keys():
                        val = getattr(data, attr)
                        if isinstance(val, torch.Tensor):
                            # Ensure tensor is on CPU
                            setattr(tensor_data, attr, val.cpu())
                        elif attr == 'mol':
                            # Store mol separately
                            mol_list.append(val)
                        # Skip other non-tensor attributes for collation
                    tensor_data_list.append(tensor_data)

                # Create batch using PyTorch Geometric's collation
                batch = Batch.from_data_list(tensor_data_list)
                # Attach mol_list to the batch
                batch.mol_list = mol_list
                return batch

            # Create data loaders (train_loader, val_loader, test_loader)
            train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True, pin_memory=True, collate_fn=custom_collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=1000, pin_memory=True, collate_fn=custom_collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=1000, pin_memory=True, collate_fn=custom_collate_fn)

            for data in [train_dataset[0], val_dataset[0], test_dataset[0]]:
                for key, val in data.__dict__.items():
                    if torch.is_tensor(val):
                        assert val.device.type == 'cpu', f"{key} is on {val.device}"

            # Create/Load libraries (using CPU data)
            print("\n  Creating libraries...")
            default_lib = None
            libs_base_path = Path('libraries') # Define base path
            libs_base_path.mkdir(parents=True, exist_ok=True) # Ensure it exists

            adaptive_lib_path = libs_base_path / f'{dataset_name.lower()}_adaptive.json'
            if adaptive_lib_path.exists():
                print("    Loading existing adaptive library...")
                try: adaptive_lib = AdaptiveSubstructureLibrary.load(str(adaptive_lib_path))
                except Exception as e: print(f"    Error loading adaptive lib: {e}"); adaptive_lib = None
            else:
                print(f"    Creating adaptive library for {dataset_name}...")
                try:
                    base_library = SubstructureLibrary(include_extended=True)
                    adaptive_lib = AdaptiveSubstructureLibrary(base_library, min_frequency=0.05, max_substructures=50)
                    adaptive_lib.fit(processed_dataset_cpu) # Use CPU data
                    adaptive_lib.save(str(adaptive_lib_path))
                except Exception as e: print(f"    Error creating adaptive lib: {e}"); adaptive_lib = None

            hierarchical_lib_path = libs_base_path / f'{dataset_name.lower()}_hierarchical.json'
            if hierarchical_lib_path.exists():
                print("    Loading existing hierarchical library...")
                try: hierarchical_lib = HierarchicalLibrary.load(str(hierarchical_lib_path))
                except Exception as e: print(f"    Error loading hierarchical lib: {e}"); hierarchical_lib = None
            else:
                print(f"    Creating hierarchical library for {dataset_name}...")
                try:
                    # You should ideally use the dataset-specific config from build_hierarchical script here
                    # Using defaults for now as fallback
                    base_library = SubstructureLibrary(include_extended=True)
                    hierarchical_lib = HierarchicalLibrary(base_library, redundancy_threshold=0.90, min_frequency=0.05)
                    hierarchical_lib.fit(processed_dataset_cpu) # Use CPU data
                    hierarchical_lib.save(str(hierarchical_lib_path))
                except Exception as e: print(f"    Error creating hierarchical lib: {e}"); hierarchical_lib = None

            # --- CHANGE 2: Added GAT, MPNN, and Graphormer to model_configs ---
            model_configs = {
                'SAGNN_default': {'library': default_lib, 'type': 'SAGNN_default'},
                'SAGNN_adaptive': {'library': adaptive_lib, 'type': 'SAGNN_adaptive'},
                'SAGNN_hierarchical': {'library': hierarchical_lib, 'type': 'SAGNN_hierarchical'},
                'GCN': {'library': None, 'type': 'GCN'},
                'GAT': {'library': None, 'type': 'GAT'},
                'MPNN': {'library': None, 'type': 'MPNN'},
                'GIN': {'library': None, 'type': 'GIN'},
                'MorganFP': {'library': None, 'type': 'MorganFP'},
                'Graphormer': {'library': None, 'type': 'Graphormer'}
            }

            dataset_results = {}

            # Test each model
            for model_name, config in model_configs.items():
                print(f"\n  Model: {model_name}")
                print(f"  {'-'*50}")

                # Skip SA-GNN variants if their library failed to load/create
                if model_name != 'SAGNN_default' and config['library'] is None and model_name.startswith('SAGNN'):
                     print("    Skipping: Required library not available.")
                     dataset_results[model_name] = None
                     continue
                # Skip MPNN if edge_dim is 0
                if model_name == 'MPNN' and edge_dim == 0:
                    print("    Skipping MPNN: Edge dimension is 0.")
                    dataset_results[model_name] = None
                    continue


                try:
                    # Create model (will be moved to self.device)
                    model = self.create_model(config['type'], dataset_info, config['library'])

                    n_params = sum(p.numel() for p in model.parameters())
                    print(f"    Parameters: {n_params:,}")

                    if hasattr(model, 'final_repr_dim'):
                        print(f"    Representation dim: {model.final_repr_dim}")
                        print(f"    Substructures: {len(model.detector.pattern_list)}")

                    print(f"    Training...")
                    model, train_time = self.train_model(model, train_loader, val_loader, task, model_name)
                    print(f"    Training time: {train_time:.2f}s")

                    print(f"    Evaluating...")
                    metrics = self.evaluate_regression(model, test_loader) if metric_type == 'regression' else self.evaluate_classification(model, test_loader)
                    if metric_type == 'regression': print(f"    MAE: {metrics.get('MAE', float('nan')):.4f}, RMSE: {metrics.get('RMSE', float('nan')):.4f}")
                    else: print(f"    ROC-AUC: {metrics.get('ROC-AUC', float('nan')):.4f}")

                    # Measure inference time (using CPU data loader for consistency, batch moved inside loop)
                    model.eval()
                    start_time = time.time()
                    num_inference_batches = 0
                    with torch.no_grad():
                        # Use a CPU version of the loader for timing, limit batches
                        cpu_test_loader = DataLoader(test_dataset, batch_size=1000, pin_memory=False) # Use Subset from split
                        for i, batch_cpu in enumerate(cpu_test_loader):
                            if i >= 5: break # Limit to 5 batches
                            batch_gpu = batch_cpu.to(self.device) # Move only the timed batch
                            if isinstance(model, SAGNN): _ = model(batch_gpu)
                            elif isinstance(model, MorganFPModel):
                                mols = [data.mol for data in batch_cpu.to_data_list() if hasattr(data,'mol')]
                                batch_gpu.mol_list = mols # Add list to GPU batch
                                _ = model(batch_gpu)
                            else: _ = model(batch_gpu)
                            num_inference_batches += 1

                    inference_time = ((time.time() - start_time) / num_inference_batches * 1000) if num_inference_batches > 0 else 0


                    metrics['training_time'] = train_time
                    metrics['inference_time_ms'] = inference_time
                    metrics['n_parameters'] = n_params

                    if hasattr(model, 'final_repr_dim'):
                        metrics['representation_dim'] = model.final_repr_dim
                        metrics['n_substructures'] = len(model.detector.pattern_list)

                    dataset_results[model_name] = metrics

                    # --- ADD VISUALIZATION CALL ---
                    if isinstance(model, SAGNN):
                        try:
                             # Get a single CPU sample from the original CPU split
                             sample_data_cpu = test_dataset.dataset[test_dataset.indices[0]]
                             if hasattr(sample_data_cpu, 'mol') and sample_data_cpu.mol is not None:
                                 visualize_importances(model, sample_data_cpu, metric_name, dataset_name)
                             else:
                                 print("    Skipping visualization: Mol object missing from sample data.")
                        except IndexError:
                             print("    Skipping visualization: Test dataset is empty.")
                        except Exception as viz_e:
                             print(f"    Error during visualization prep: {viz_e}")
                    # ---

                except Exception as e:
                    print(f"    Error running model {model_name}: {e}")
                    import traceback; traceback.print_exc()
                    dataset_results[model_name] = None

            # Store results
            self.results['library_comparison'][dataset_name] = dataset_results

            # Print summary for this dataset
            self._print_dataset_summary(dataset_name, dataset_results, metric_type)

        # Save all results
        print(f"\n{'='*70}")
        print("SAVING RESULTS")
        print('='*70)

        with open(self.results_dir / 'hierarchical_benchmark_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"Results saved to {self.results_dir / 'hierarchical_benchmark_results.json'}")

        # Generate final summary
        self.generate_summary()

        return self.results

    def _print_dataset_summary(self, dataset_name, results, metric_type):
        """Print summary for one dataset, sorted by performance"""
        print(f"\n  {'='*50}")
        print(f"  {dataset_name} SUMMARY")
        print(f"  {'='*50}")

        if metric_type == 'regression':
            print(f"  {'Model':<20} {'RMSE':<10} {'Params':<12} {'SubStructs'}")
            print(f"  {'-'*50}")
            # Sort by RMSE (lower is better)
            for model_name, metrics in sorted(results.items(), key=lambda item: (item[1] or {}).get('RMSE', float('inf')) if item[1] is not None else float('inf')):
                 if metrics:
                    rmse = metrics.get('RMSE', float('nan'))
                    params = metrics.get('n_parameters', 0)
                    substrs = metrics.get('n_substructures', '-')
                    print(f"  {model_name:<20} {rmse:<10.4f} {params:<12,} {substrs}")
        else:
            print(f"  {'Model':<20} {'ROC-AUC':<10} {'Params':<12} {'SubStructs'}")
            print(f"  {'-'*50}")
            # Sort by ROC-AUC (higher is better)
            for model_name, metrics in sorted(results.items(), key=lambda item: (item[1] or {}).get('ROC-AUC', float('-inf')) if item[1] is not None else float('-inf'), reverse=True):
                 if metrics:
                    auc = metrics.get('ROC-AUC', float('nan'))
                    params = metrics.get('n_parameters', 0)
                    substrs = metrics.get('n_substructures', '-')
                    print(f"  {model_name:<20} {auc:<10.4f} {params:<12,} {substrs}")


    def generate_summary(self):
        """Generate comprehensive summary with corrected best model logic"""
        print(f"\n{'='*70}")
        print("COMPREHENSIVE BENCHMARK SUMMARY")
        print(f"{'='*70}")

        for dataset_name, models in self.results['library_comparison'].items():
            print(f"\n{dataset_name}:")

            best_model = None
            # Check if metrics exist for any model before proceeding
            valid_metrics = [m for m in models.values() if m is not None]
            if not valid_metrics:
                print("  No valid results for this dataset.")
                continue
            
            # Check for 'RMSE' in the first valid metric dict
            is_regression = 'RMSE' in valid_metrics[0] 

            best_score = float('inf') if is_regression else float('-inf')

            for model_name, metrics in models.items():
                if not metrics: continue # Skip if model run failed

                if is_regression:
                    score = metrics.get('RMSE', float('inf'))
                    if not np.isnan(score) and score < best_score:
                        best_model, best_score = model_name, score
                else:
                    score = metrics.get('ROC-AUC', float('-inf'))
                    if not np.isnan(score) and score > best_score:
                        best_model, best_score = model_name, score

            if best_model is None:
                print(f"  Best model: None (All results were NaN)")
            else:
                print(f"  Best model: {best_model} ({'RMSE' if is_regression else 'ROC-AUC'}: {best_score:.4f})")


            # Compare hierarchical vs Default
            hier_metrics = models.get('SAGNN_hierarchical')
            default_metrics = models.get('SAGNN_default')

            if hier_metrics and default_metrics and hier_metrics.get('n_parameters') and default_metrics.get('n_parameters'):
                if is_regression:
                    hier_score = hier_metrics.get('RMSE', float('inf'))
                    def_score = default_metrics.get('RMSE', float('inf'))
                    if not np.isnan(hier_score) and not np.isnan(def_score) and def_score > 1e-9:
                        improvement = (def_score - hier_score) / def_score * 100
                        print(f"  Hierarchical vs Default: {improvement:+.1f}% RMSE improvement")
                    else:
                        print(f"  Hierarchical vs Default: RMSE improvement N/A (NaN or ~0 score)")
                else: # Classification
                    hier_score = hier_metrics.get('ROC-AUC', float('-inf'))
                    def_score = default_metrics.get('ROC-AUC', float('-inf'))
                    
                    if not np.isnan(hier_score) and not np.isnan(def_score):
                        if def_score > 0.51: # Check if default learned something
                            improvement = (hier_score - def_score) / def_score * 100
                            print(f"  Hierarchical vs Default: {improvement:+.1f}% ROC-AUC improvement")
                        elif hier_score > def_score:
                             print(f"  Hierarchical vs Default: Hierarchical improved over failing Default (AUC {hier_score:.4f} vs {def_score:.4f})")
                        else:
                             print(f"  Hierarchical vs Default: No significant improvement (Both AUC <= 0.51 or NaN)")
                    else:
                        print(f"  Hierarchical vs Default: ROC-AUC improvement N/A (NaN score)")


                # Parameter and Substructure reduction (safe comparison)
                param_reduction = (default_metrics['n_parameters'] - hier_metrics['n_parameters']) / default_metrics['n_parameters'] * 100
                print(f"  Parameter reduction: {param_reduction:.1f}%")

                substruct_reduction = (default_metrics['n_substructures'] - hier_metrics['n_substructures']) / default_metrics['n_substructures'] * 100
                print(f"  Substructure reduction: {substruct_reduction:.1f}%")
            elif hier_metrics:
                 print("  Default SA-GNN results missing for comparison.")
            elif default_metrics:
                 print("  Hierarchical SA-GNN results missing for comparison.")


        print(f"\n{'='*70}")


if __name__ == '__main__':
    # Run complete hierarchical benchmark
    benchmark = HierarchicalBenchmark(
        # --- SET YOUR DESIRED GPU HERE ---
        device='cuda:3' if torch.cuda.is_available() else 'cpu'
    )

    results = benchmark.run_hierarchical_comparison()

    print("\n" + "="*70)
    print("HIERARCHICAL BENCHMARK COMPLETE!")
    print("="*70)