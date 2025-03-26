# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_scatter import scatter_mean, scatter_add

class ImprovedGraphSRModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, embedding_dim, output_dim, dropout=0.5, num_heads=4):
        super(ImprovedGraphSRModel, self).__init__()
        self.embedding_dim = embedding_dim

        # Graph convolution layers
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels // 2)
        self.conv3 = SAGEConv(hidden_channels // 2, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        # Node-level attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Decoder for final output
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 8192),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(8192, output_dim),
            nn.Sigmoid()
        )

        # Multi-head attention and associated layers
        self.mha = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        # Feed-forward network for transformer-style processing
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )

        # Edge attention mechanism
        self.edge_attention = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Global context for graph-level embedding
        self.global_pool = nn.Parameter(torch.zeros(1, embedding_dim))
        nn.init.xavier_uniform_(self.global_pool)

        # Gating mechanism for combining node features
        self.gating = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, batch):
        # Apply graph convolutions
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = self.dropout(x)

        # Compute node-level attention weights
        attn_scores = self.attention(x)
        attn_weights = torch.softmax(attn_scores, dim=0)

        # Compute edge weights based on node features
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col]], dim=1)
        edge_weights = self.edge_attention(edge_features).squeeze(-1)

        # Compute per-node edge weight influence (average edge weight per node)
        # This assigns each node a weight based on the average edge weight of its incident edges
        edge_weight_per_node = (scatter_add(edge_weights, row, dim=0, dim_size=x.shape[0]) + 
                                scatter_add(edge_weights, col, dim=0, dim_size=x.shape[0])) / 2

        # Since all graphs have the same number of nodes, compute the number of nodes per graph
        batch_size = torch.max(batch).item() + 1  # Number of graphs in the batch
        num_nodes_per_graph = (batch == 0).sum().item()  # Assuming all graphs have the same number of nodes

        # Reshape node features for multi-head attention
        # Shape: (num_nodes_per_graph, batch_size, embedding_dim)
        node_features = x.view(batch_size, num_nodes_per_graph, self.embedding_dim).transpose(0, 1)

        # Apply multi-head attention
        attn_output, _ = self.mha(node_features, node_features, node_features)
        
        # Add residual connection and normalize
        node_features = self.layer_norm1(node_features + attn_output)

        # Apply feed-forward network and another residual connection
        ffn_output = self.ffn(node_features)
        node_features = self.layer_norm2(node_features + ffn_output)

        # Reshape back to (num_nodes_total, embedding_dim) for further processing
        enhanced_x = node_features.transpose(0, 1).reshape(-1, self.embedding_dim)

        # Apply gating mechanism to combine original and enhanced node features
        gate_input = torch.cat([x, enhanced_x], dim=1)
        gate = self.gating(gate_input)
        x_combined = gate * enhanced_x + (1 - gate) * x

        # Apply edge weights to node features before global aggregation
        # Scale node features by their corresponding edge weights
        weighted_x_combined = x_combined * edge_weight_per_node.unsqueeze(-1)

        # Compute global graph embedding with global context
        global_context = self.global_pool.expand(batch_size, -1)
        global_graph_emb = scatter_mean(weighted_x_combined, batch, dim=0)  # Use weighted node features
        graph_embedding_with_global = global_graph_emb + global_context

        # Compute final graph embedding using attention weights
        final_graph_embedding = scatter_mean(weighted_x_combined * attn_weights, batch, dim=0)

        # Combine embeddings for final prediction
        final_embedding = (final_graph_embedding + graph_embedding_with_global) / 2

        # Decode to get the final output
        pred_vector = self.decoder(final_embedding)

        return pred_vector, x_combined