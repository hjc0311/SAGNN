"""
GNN backbone with edge features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class EdgeConv(MessagePassing):
    """
    Message passing layer that incorporates edge features
    """
    
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int):
        super().__init__(aggr='add')
        
        # Message function: combines source, target, and edge
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + edge_dim, out_channels * 2),
            nn.BatchNorm1d(out_channels * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_channels * 2, out_channels)
        )
        
        # Update function
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x, edge_index, edge_attr):
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Add self-loop edge features (zeros)
        if edge_attr is not None:
            self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1), 
                                        device=edge_attr.device)
            edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        else:
            edge_attr = torch.zeros(edge_index.size(1), 1, device=x.device)
        
        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # Update node features
        out = self.update_mlp(torch.cat([x, out], dim=1))
        
        return out
    
    def message(self, x_i, x_j, edge_attr):
        # x_i: target node [n_edges, in_channels]
        # x_j: source node [n_edges, in_channels]
        # edge_attr: edge features [n_edges, edge_dim]
        
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(msg_input)


class SAGNNBackbone(nn.Module):
    """
    GNN backbone with edge features and residual connections
    """
    
    def __init__(self, 
                 node_dim: int, 
                 edge_dim: int, 
                 hidden_dim: int, 
                 n_layers: int,
                 dropout: float = 0.1):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Initial node embedding
        self.node_embedding = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Message passing layers
        self.convs = nn.ModuleList([
            EdgeConv(hidden_dim, hidden_dim, edge_dim)
            for _ in range(n_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Node features [n_nodes, node_dim]
            edge_index: Edge connectivity [2, n_edges]
            edge_attr: Edge features [n_edges, edge_dim]
            
        Returns:
            Updated node features [n_nodes, hidden_dim]
        """
        # Initial embedding
        h = self.node_embedding(x)
        
        # Message passing with residual connections
        for i, (conv, norm) in enumerate(zip(self.convs, self.layer_norms)):
            h_new = conv(h, edge_index, edge_attr)
            h_new = norm(h_new)
            h_new = self.dropout(h_new)
            
            # Residual connection
            h = h + h_new
        
        return h