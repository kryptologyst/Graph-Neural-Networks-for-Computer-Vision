#!/usr/bin/env python3
"""
Project 429: Graph Neural Networks for Computer Vision

This is the original implementation that has been modernized into a full project.
The modernized version is now available in the project structure with:
- Multiple GNN architectures (GCN, GraphSAGE, GAT, GIN, Transformer)
- Comprehensive evaluation and visualization tools
- Interactive Streamlit demo
- Production-ready code with type hints and testing

For the modernized version, see:
- scripts/train.py for training
- demo/app.py for interactive demo
- example.py for quick start

Original implementation preserved below for reference.
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import SuperpixelsSLIC
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures, AddSelfLoops


def main():
    """Original implementation with modern improvements."""
    print("🧠 Graph Neural Networks for Computer Vision - Original Implementation")
    print("=" * 70)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # 1. Load CIFAR10 Superpixel dataset with modern transforms
    print("Loading CIFAR10 Superpixel dataset...")
    
    # Add modern transforms
    transform = torch.nn.Sequential(
        NormalizeFeatures(),
        AddSelfLoops()
    )
    
    dataset = SuperpixelsSLIC(
        root='data/superpixels', 
        name='CIFAR10', 
        split='train', 
        transform=transform,
        slic_segments=75,
        slic_compactness=10.0
    )
    
    dataset = dataset.shuffle()
    train_dataset = dataset[:500]
    test_dataset = dataset[500:600]
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test samples")
    print(f"Node features: {dataset.num_node_features}, Classes: {dataset.num_classes}")
    
    # 2. Define improved GCN model
    class SuperpixelGCN(torch.nn.Module):
        def __init__(self, in_channels: int, hidden_channels: int, num_classes: int, dropout: float = 0.5):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
            self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
            self.dropout = torch.nn.Dropout(dropout)
            self.lin1 = torch.nn.Linear(hidden_channels, 64)
            self.lin2 = torch.nn.Linear(64, num_classes)
            
        def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
            # First GCN layer
            x = self.conv1(x, edge_index)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.dropout(x)
            
            # Second GCN layer
            x = self.conv2(x, edge_index)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.dropout(x)
            
            # Global pooling and classification
            x = global_mean_pool(x, batch)
            x = F.relu(self.lin1(x))
            x = self.dropout(x)
            return F.log_softmax(self.lin2(x), dim=1)
    
    # 3. Model setup with modern device handling
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() 
                         else 'cpu')
    
    model = SuperpixelGCN(
        in_channels=dataset.num_node_features, 
        hidden_channels=64, 
        num_classes=dataset.num_classes,
        dropout=0.5
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    loss_fn = torch.nn.NLLLoss()
    
    print(f"Model created on device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 4. Improved training function
    def train() -> float:
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(out, batch.y)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
        
        return total_loss / len(train_loader), correct / total
    
    # 5. Improved evaluation function
    def test() -> tuple[float, float]:
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = loss_fn(out, batch.y)
                
                total_loss += loss.item()
                pred = out.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
        
        return total_loss / len(test_loader), correct / total
    
    # 6. Enhanced training loop
    print("\\nStarting training...")
    print("-" * 50)
    
    best_acc = 0
    for epoch in range(1, 21):
        train_loss, train_acc = train()
        test_loss, test_acc = test()
        scheduler.step()
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        print(f"Epoch {epoch:02d}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
    
    print("-" * 50)
    print(f"Training completed! Best test accuracy: {best_acc:.4f}")
    
    print("\\n" + "=" * 70)
    print("✅ Original implementation completed!")
    print("\\nFor the modernized version with advanced features:")
    print("• Run: python scripts/train.py --compare_models")
    print("• Demo: streamlit run demo/app.py")
    print("• Example: python example.py")


if __name__ == "__main__":
    main()