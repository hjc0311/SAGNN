"""
Baseline models for comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool, global_add_pool
from torch_geometric.nn import MessagePassing
from rdkit.Chem import AllChem
from torch_geometric.utils import add_self_loops, degree, to_dense_batch


class GCN(nn.Module):
    """Standard Graph Convolutional Network"""
    
    def __init__(self, 
                 node_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 n_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_dim, hidden_dim))
        
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        x = global_mean_pool(x, batch)
        return self.predictor(x)


class GAT(nn.Module):
    """Graph Attention Network"""
    
    def __init__(self,
                 node_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 n_layers: int = 3,
                 heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(node_dim, hidden_dim, heads=heads, concat=True))
        
        for _ in range(n_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True))
        
        # Last layer with averaging
        self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=False))
        
        self.dropout = nn.Dropout(dropout)
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = self.dropout(x)
        
        x = global_mean_pool(x, batch)
        return self.predictor(x)


class MPNNConv(MessagePassing):
    """Message Passing Neural Network layer with edge features"""
    
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__(aggr='add')
        
        self.edge_network = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim * hidden_dim)
        )
        
        self.node_network = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr):
        edge_embedding = self.edge_network(edge_attr)
        edge_embedding = edge_embedding.view(-1, x_j.size(1), self.edge_network[-1].out_features // x_j.size(1))
        return torch.matmul(x_j.unsqueeze(1), edge_embedding).squeeze(1)
    
    def update(self, aggr_out):
        return self.node_network(aggr_out)


class MPNN(nn.Module):
    """Message Passing Neural Network"""
    
    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 n_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        
        self.convs = nn.ModuleList([
            MPNNConv(hidden_dim, edge_dim, hidden_dim)
            for _ in range(n_layers)
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, data):
        x = self.node_embedding(data.x)
        edge_index, edge_attr = data.edge_index, data.edge_attr
        
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        x = global_mean_pool(x, data.batch)
        return self.predictor(x)


class MorganFPModel(nn.Module):
    """
    Baseline using Morgan Fingerprints
    Default: 4096 bits, radius 3 (as specified)
    """
    
    def __init__(self,
                 fp_size: int = 4096,
                 radius: int = 3,
                 hidden_dim: int = 512,
                 output_dim: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        
        self.fp_size = fp_size
        self.radius = radius
        
        self.predictor = nn.Sequential(
            nn.Linear(fp_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def compute_fingerprint(self, mol):
        """Compute Morgan fingerprint for a single molecule"""
        if mol is None:
            # Handle cases where RDKit failed to parse the molecule
            return torch.zeros(self.fp_size, dtype=torch.float32)
            
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, 
            radius=self.radius, 
            nBits=self.fp_size
        )
        return torch.tensor(list(fp), dtype=torch.float32)
    
    def forward(self, data):
        """
        Args:
            data: Must have 'mol' or 'mol_list' attribute
        """
        device = next(self.parameters()).device
        
        # Handle single molecule or batch
        if hasattr(data, 'mol_list'):
            mols = data.mol_list
        elif hasattr(data, 'mol'):
            mols = [data.mol]
        else:
            raise ValueError("Data must have 'mol' or 'mol_list' attribute")
        
        # Compute fingerprints
        fps = torch.stack([
            self.compute_fingerprint(mol) for mol in mols
        ]).to(device)
        
        return self.predictor(fps)


class GIN(nn.Module):
    """Graph Isomorphism Network (GIN)"""
    
    def __init__(self, 
                 node_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 n_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        mlp1 = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(mlp1, train_eps=True))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Subsequent layers
        for _ in range(n_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # GIN typically uses sum pooling
        x = global_add_pool(x, batch) 
        return self.predictor(x)
    

class Graphormer(nn.Module):
    """
    Simplified Graphormer-like model for baseline comparison.
    Uses a TransformerEncoder over node features, with degree
    encoding added. Does not use the full spatial/edge attention biases
    from the paper, as they are non-trivial to implement with
    standard nn.TransformerEncoder and PyG batching.
    
    This implementation ignores edge features, similar to GCN/GAT.
    """
    def __init__(self,
                 node_dim: int,
                 edge_dim: int, # Kept for API compatibility, but unused
                 hidden_dim: int = 128,
                 output_dim: int = 1,
                 n_layers: int = 4,
                 n_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        
        # Node and degree embeddings
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        # Max degree 50 is a common assumption
        self.degree_encoder = nn.Embedding(50, hidden_dim)  
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True  # Important: [B, N, H]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output predictor
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 1. Get Degree Encoding (a key Graphormer feature)
        # We add self-loops for degree calculation, as a node's
        # relation to itself is implicitly handled by attention.
        edge_index_sl, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        degree_vec = degree(edge_index_sl[0], num_nodes=x.size(0)).long()
        # Clamp at max degree 49
        degree_vec = torch.clamp(degree_vec, max=49) 
        
        degree_bias = self.degree_encoder(degree_vec) # [n_nodes, hidden_dim]
        
        # 2. Embed node features
        x_emb = self.node_embedding(x)  # [n_nodes, hidden_dim]
        
        # 3. Add degree bias to node features
        x = x_emb + degree_bias
        
        # 4. Convert to dense, padded batch for Transformer
        # x_padded: [B, N_max, H]
        # mask: [B, N_max] (boolean, True where node exists)
        x_padded, mask = to_dense_batch(x, batch)
        
        # 5. Create padding mask for transformer
        # padding_mask: [B, N_max] (boolean, True where node is PADDING)
        padding_mask = ~mask
        
        # 6. Transformer forward pass
        # Input: [B, N_max, H]
        # Mask: [B, N_max]
        transformer_out = self.transformer(
            x_padded,
            src_key_padding_mask=padding_mask
        ) # Output: [B, N_max, H]
        
        # 7. Pool to get graph-level representation
        # We do a masked average pool
        transformer_out[padding_mask] = 0.0 # Zero out padded nodes
        graph_repr = transformer_out.sum(dim=1) # [B, H]
        n_nodes_per_graph = mask.sum(dim=1).unsqueeze(-1) # [B, 1]
        graph_repr = graph_repr / n_nodes_per_graph.clamp(min=1) # [B, H]
        
        # 8. Final prediction
        return self.predictor(graph_repr)