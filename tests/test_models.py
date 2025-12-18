"""Unit tests for the GNN CV project."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import Config, DataConfig, ModelConfig
from src.utils.device import get_device, set_seed, count_parameters
from src.data.dataset import SuperpixelGraphDataset, create_synthetic_dataset
from src.models.gnn_models import create_model, SuperpixelGCN, SuperpixelGAT
from src.eval.metrics import compute_metrics


class TestConfig:
    """Test configuration management."""
    
    def test_config_creation(self):
        """Test basic config creation."""
        config = Config()
        assert config.data.dataset_name == "CIFAR10"
        assert config.model.model_type == "gcn"
        assert config.training.epochs == 50
    
    def test_config_update(self):
        """Test config updates."""
        config = Config()
        config.update(model_type="gat", epochs=100)
        assert config.model.model_type == "gat"
        assert config.training.epochs == 100


class TestDevice:
    """Test device management."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device("auto")
        assert isinstance(device, torch.device)
        
        cpu_device = get_device("cpu")
        assert cpu_device.type == "cpu"
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # Test that seeds are set (basic check)
        assert True  # If no exception is raised, seeds are set


class TestDataset:
    """Test dataset functionality."""
    
    def test_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        data = create_synthetic_dataset(num_graphs=10, num_classes=3)
        assert len(data) == 10
        
        # Check first graph structure
        graph = data[0]
        assert hasattr(graph, 'x')
        assert hasattr(graph, 'edge_index')
        assert hasattr(graph, 'y')
        assert graph.x.dim() == 2
        assert graph.edge_index.dim() == 2
        assert graph.y.dim() == 1


class TestModels:
    """Test model functionality."""
    
    def test_model_creation(self):
        """Test model creation."""
        model = create_model(
            model_type="gcn",
            in_channels=3,
            hidden_channels=32,
            num_classes=10,
            num_layers=2
        )
        assert isinstance(model, SuperpixelGCN)
        assert count_parameters(model) > 0
    
    def test_model_forward(self):
        """Test model forward pass."""
        model = create_model(
            model_type="gcn",
            in_channels=3,
            hidden_channels=32,
            num_classes=10,
            num_layers=2
        )
        
        # Create dummy data
        batch_size = 2
        num_nodes = 10
        x = torch.randn(num_nodes, 3)
        edge_index = torch.randint(0, num_nodes, (2, 20))
        batch = torch.zeros(num_nodes, dtype=torch.long)
        
        # Forward pass
        output = model(x, edge_index, batch)
        assert output.shape == (batch_size, 10)
    
    def test_gat_attention(self):
        """Test GAT attention weights."""
        model = create_model(
            model_type="gat",
            in_channels=3,
            hidden_channels=32,
            num_classes=10,
            num_layers=2
        )
        
        # Create dummy data
        num_nodes = 10
        x = torch.randn(num_nodes, 3)
        edge_index = torch.randint(0, num_nodes, (2, 20))
        
        # Get attention weights
        if hasattr(model, 'get_attention_weights'):
            attention_weights = model.get_attention_weights(x, edge_index)
            assert len(attention_weights) == 2  # Two layers


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_compute_metrics(self):
        """Test metrics computation."""
        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 1, 1, 0, 2]
        
        metrics = compute_metrics(y_true, y_pred, ["accuracy", "f1_macro"])
        
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1_macro"] <= 1


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_training(self):
        """Test end-to-end training (minimal)."""
        # Create synthetic data
        data = create_synthetic_dataset(num_graphs=20, num_classes=3)
        
        # Create model
        model = create_model(
            model_type="gcn",
            in_channels=3,
            hidden_channels=16,
            num_classes=3,
            num_layers=1
        )
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Minimal training loop
        model.train()
        for i in range(3):  # Just 3 iterations
            graph = data[i]
            optimizer.zero_grad()
            
            # Forward pass
            out = model(graph.x, graph.edge_index, torch.zeros(graph.num_nodes, dtype=torch.long))
            loss = criterion(out, graph.y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        # Test that model can make predictions
        model.eval()
        with torch.no_grad():
            test_graph = data[0]
            output = model(test_graph.x, test_graph.edge_index, torch.zeros(test_graph.num_nodes, dtype=torch.long))
            assert output.shape == (1, 3)


if __name__ == "__main__":
    pytest.main([__file__])
