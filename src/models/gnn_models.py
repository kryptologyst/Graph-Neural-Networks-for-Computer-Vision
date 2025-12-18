"""Graph Neural Network models for superpixel-based image classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GraphSAGE, GATConv, GINConv, 
    global_mean_pool, global_max_pool, global_add_pool,
    Set2Set, TransformerConv, GINEConv
)
from torch_geometric.nn.pool import TopKPooling, SAGPooling
from typing import Optional, List, Dict, Any
import math


class BaseGNNModel(nn.Module):
    """Base class for GNN models."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        activation: str = "relu",
        use_batch_norm: bool = True,
        use_residual: bool = False,
        pooling: str = "mean"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        self.pooling = pooling
        
        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "elu":
            self.activation = F.elu
        else:
            self.activation = F.relu
        
        # Batch normalization layers
        if use_batch_norm:
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)
            ])
        
        # Pooling layer
        if pooling == "mean":
            self.pool = global_mean_pool
        elif pooling == "max":
            self.pool = global_max_pool
        elif pooling == "sum":
            self.pool = global_add_pool
        elif pooling == "set2set":
            self.pool = Set2Set(hidden_channels, processing_steps=3)
        else:
            self.pool = global_mean_pool
        
        # Classifier
        if pooling == "set2set":
            self.classifier = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 2, num_classes)
            )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError


class SuperpixelGCN(BaseGNNModel):
    """Graph Convolutional Network for superpixel classification."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(self.in_channels, self.hidden_channels))
        
        for _ in range(self.num_layers - 1):
            self.convs.append(GCNConv(self.hidden_channels, self.hidden_channels))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for i, conv in enumerate(self.convs):
            residual = x if self.use_residual and i > 0 else None
            
            x = conv(x, edge_index)
            
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            if residual is not None and x.size() == residual.size():
                x = x + residual
        
        # Global pooling
        x = self.pool(x, batch)
        
        # Classification
        return self.classifier(x)


class SuperpixelGraphSAGE(BaseGNNModel):
    """GraphSAGE for superpixel classification."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.convs.append(GraphSAGE(self.in_channels, self.hidden_channels, num_layers=1))
        
        for _ in range(self.num_layers - 1):
            self.convs.append(GraphSAGE(self.hidden_channels, self.hidden_channels, num_layers=1))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for i, conv in enumerate(self.convs):
            residual = x if self.use_residual and i > 0 else None
            
            x = conv(x, edge_index)
            
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            if residual is not None and x.size() == residual.size():
                x = x + residual
        
        # Global pooling
        x = self.pool(x, batch)
        
        # Classification
        return self.classifier(x)


class SuperpixelGAT(BaseGNNModel):
    """Graph Attention Network for superpixel classification."""
    
    def __init__(self, heads: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.heads = heads
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(self.in_channels, self.hidden_channels // heads, heads=heads))
        
        for _ in range(self.num_layers - 1):
            self.convs.append(GATConv(self.hidden_channels, self.hidden_channels // heads, heads=heads))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for i, conv in enumerate(self.convs):
            residual = x if self.use_residual and i > 0 else None
            
            x = conv(x, edge_index)
            
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            if residual is not None and x.size() == residual.size():
                x = x + residual
        
        # Global pooling
        x = self.pool(x, batch)
        
        # Classification
        return self.classifier(x)
    
    def get_attention_weights(self, x: torch.Tensor, edge_index: torch.Tensor) -> List[torch.Tensor]:
        """Get attention weights for visualization."""
        attention_weights = []
        
        for conv in self.convs:
            _, att = conv(x, edge_index, return_attention_weights=True)
            attention_weights.append(att)
            x = conv(x, edge_index)
            x = self.activation(x)
        
        return attention_weights


class SuperpixelGIN(BaseGNNModel):
    """Graph Isomorphism Network for superpixel classification."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # GIN layers
        self.convs = nn.ModuleList()
        
        # First layer
        nn1 = nn.Sequential(
            nn.Linear(self.in_channels, self.hidden_channels),
            nn.ReLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels)
        )
        self.convs.append(GINConv(nn1))
        
        # Subsequent layers
        for _ in range(self.num_layers - 1):
            nn_layer = nn.Sequential(
                nn.Linear(self.hidden_channels, self.hidden_channels),
                nn.ReLU(),
                nn.Linear(self.hidden_channels, self.hidden_channels)
            )
            self.convs.append(GINConv(nn_layer))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for i, conv in enumerate(self.convs):
            residual = x if self.use_residual and i > 0 else None
            
            x = conv(x, edge_index)
            
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            if residual is not None and x.size() == residual.size():
                x = x + residual
        
        # Global pooling
        x = self.pool(x, batch)
        
        # Classification
        return self.classifier(x)


class SuperpixelTransformer(BaseGNNModel):
    """Graph Transformer for superpixel classification."""
    
    def __init__(self, heads: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.heads = heads
        
        # Transformer layers
        self.convs = nn.ModuleList()
        self.convs.append(TransformerConv(self.in_channels, self.hidden_channels, heads=heads))
        
        for _ in range(self.num_layers - 1):
            self.convs.append(TransformerConv(self.hidden_channels, self.hidden_channels, heads=heads))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for i, conv in enumerate(self.convs):
            residual = x if self.use_residual and i > 0 else None
            
            x = conv(x, edge_index)
            
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            if residual is not None and x.size() == residual.size():
                x = x + residual
        
        # Global pooling
        x = self.pool(x, batch)
        
        # Classification
        return self.classifier(x)


def create_model(model_type: str, **kwargs) -> BaseGNNModel:
    """
    Factory function to create GNN models.
    
    Args:
        model_type: Type of model ("gcn", "graphsage", "gat", "gin", "transformer")
        **kwargs: Model configuration parameters
        
    Returns:
        BaseGNNModel: Instantiated model
    """
    model_classes = {
        "gcn": SuperpixelGCN,
        "graphsage": SuperpixelGraphSAGE,
        "gat": SuperpixelGAT,
        "gin": SuperpixelGIN,
        "transformer": SuperpixelTransformer
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model_classes[model_type](**kwargs)
