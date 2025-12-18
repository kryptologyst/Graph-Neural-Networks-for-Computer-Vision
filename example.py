"""Example script demonstrating the modernized GNN CV project."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

import torch
import numpy as np
from src.utils.config import Config
from src.utils.device import get_device, set_seed
from src.data.dataset import SuperpixelGraphDataset, create_synthetic_dataset
from src.models.gnn_models import create_model
from src.train.trainer import Trainer
from src.eval.metrics import compute_metrics, create_model_leaderboard


def main():
    """Main example function."""
    print("🧠 Graph Neural Networks for Computer Vision - Example")
    print("=" * 60)
    
    # Setup
    config = Config()
    device = get_device("auto")
    set_seed(42, deterministic=True)
    
    print(f"Using device: {device}")
    print(f"Model type: {config.model.model_type}")
    
    # Create synthetic dataset for quick demo
    print("\\nCreating synthetic dataset...")
    synthetic_data = create_synthetic_dataset(
        num_graphs=100,
        num_classes=5,
        avg_nodes=30,
        avg_edges=50,
        num_features=3
    )
    
    print(f"Created {len(synthetic_data)} synthetic graphs")
    
    # Create model
    print("\\nCreating model...")
    model = create_model(
        model_type=config.model.model_type,
        in_channels=3,
        hidden_channels=32,
        num_classes=5,
        num_layers=2,
        dropout=0.3
    )
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Quick training demo
    print("\\nRunning quick training demo...")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(5):
        total_loss = 0
        correct = 0
        total = 0
        
        for i, graph in enumerate(synthetic_data[:20]):  # Use first 20 graphs
            graph = graph.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            out = model(graph.x, graph.edge_index, torch.zeros(graph.num_nodes, dtype=torch.long, device=device))
            loss = criterion(out, graph.y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = torch.argmax(out, dim=1)
            correct += (pred == graph.y).sum().item()
            total += 1
        
        accuracy = correct / total
        avg_loss = total_loss / total
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
    
    # Evaluation
    print("\\nEvaluating model...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for graph in synthetic_data[20:30]:  # Use next 10 graphs for testing
            graph = graph.to(device)
            out = model(graph.x, graph.edge_index, torch.zeros(graph.num_nodes, dtype=torch.long, device=device))
            pred = torch.argmax(out, dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(graph.y.cpu().numpy())
    
    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, ["accuracy", "f1_macro"])
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1-Macro: {metrics['f1_macro']:.4f}")
    
    print("\\n✅ Example completed successfully!")
    print("\\nTo run the full training pipeline:")
    print("python scripts/train.py --model gcn --epochs 50")
    print("\\nTo launch the interactive demo:")
    print("streamlit run demo/app.py")


if __name__ == "__main__":
    main()
