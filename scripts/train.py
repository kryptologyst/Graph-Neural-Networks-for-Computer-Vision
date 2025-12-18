"""Main training script for Graph Neural Networks for Computer Vision."""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from src.utils.config import Config
from src.utils.device import get_device, set_seed
from src.data.dataset import SuperpixelGraphDataset
from src.models.gnn_models import create_model
from src.train.trainer import Trainer
from src.eval.metrics import create_model_leaderboard, plot_training_history


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train GNN models for superpixel-based image classification")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "graphsage", "gat", "gin", "transformer"], help="Model type")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--hidden_channels", type=int, default=64, help="Hidden channels")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate, don't train")
    parser.add_argument("--compare_models", action="store_true", help="Compare all model types")
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = Config.from_yaml(args.config)
    else:
        config = Config()
    
    # Override config with command line arguments
    config.model.model_type = args.model
    config.training.epochs = args.epochs
    config.data.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.model.hidden_channels = args.hidden_channels
    config.system.device = args.device
    config.system.seed = args.seed
    
    # Set up device and seeding
    device = get_device(config.system.device)
    set_seed(config.system.seed, config.system.deterministic)
    
    print(f"Using device: {device}")
    print(f"Configuration: {config.model.model_type} model with {config.training.epochs} epochs")
    
    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("assets", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    if args.compare_models:
        # Compare all model types
        model_types = ["gcn", "graphsage", "gat", "gin", "transformer"]
        results = {}
        
        for model_type in model_types:
            print(f"\\n{'='*60}")
            print(f"Training {model_type.upper()} model")
            print(f"{'='*60}")
            
            # Update config for this model
            config.model.model_type = model_type
            
            # Load dataset
            dataset = SuperpixelGraphDataset(
                dataset_name=config.data.dataset_name,
                data_root=config.data.data_root,
                superpixel_segments=config.data.superpixel_segments,
                superpixel_compactness=config.data.superpixel_compactness,
                normalize=config.data.normalize
            )
            
            train_loader, val_loader, test_loader = dataset.get_data_loaders(
                train_size=config.data.train_size,
                val_size=config.data.val_size,
                test_size=config.data.test_size,
                batch_size=config.data.batch_size,
                num_workers=config.data.num_workers,
                random_state=config.system.seed
            )
            
            # Create model
            model = create_model(
                model_type=config.model.model_type,
                in_channels=dataset.num_node_features,
                hidden_channels=config.model.hidden_channels,
                num_classes=dataset.num_classes,
                num_layers=config.model.num_layers,
                dropout=config.model.dropout,
                activation=config.model.activation,
                use_batch_norm=config.model.use_batch_norm,
                use_residual=config.model.use_residual,
                pooling=config.model.pooling
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                config=config,
                device=device
            )
            
            # Train model
            training_results = trainer.train()
            
            # Store results
            results[model_type] = training_results['test_metrics']
            
            # Save training history plot
            plot_training_history(
                training_results['history'],
                save_path=f"assets/{model_type}_training_history.png"
            )
        
        # Create leaderboard
        create_model_leaderboard(results)
        
    else:
        # Train single model
        # Load dataset
        dataset = SuperpixelGraphDataset(
            dataset_name=config.data.dataset_name,
            data_root=config.data.data_root,
            superpixel_segments=config.data.superpixel_segments,
            superpixel_compactness=config.data.superpixel_compactness,
            normalize=config.data.normalize
        )
        
        print(f"Dataset info: {dataset.get_dataset_info()}")
        
        train_loader, val_loader, test_loader = dataset.get_data_loaders(
            train_size=config.data.train_size,
            val_size=config.data.val_size,
            test_size=config.data.test_size,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            random_state=config.system.seed
        )
        
        # Create model
        model = create_model(
            model_type=config.model.model_type,
            in_channels=dataset.num_node_features,
            hidden_channels=config.model.hidden_channels,
            num_classes=dataset.num_classes,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            activation=config.model.activation,
            use_batch_norm=config.model.use_batch_norm,
            use_residual=config.model.use_residual,
            pooling=config.model.pooling
        )
        
        print(f"Model: {model.__class__.__name__}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        if args.eval_only:
            # Load checkpoint and evaluate
            checkpoint_path = f"checkpoints/best_model_{config.model.model_type}.pt"
            if os.path.exists(checkpoint_path):
                from src.utils.device import load_checkpoint
                load_checkpoint(model, None, checkpoint_path)
                
                trainer = Trainer(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    config=config,
                    device=device
                )
                
                test_metrics = trainer.test()
                print(f"Test metrics: {test_metrics}")
            else:
                print(f"Checkpoint not found: {checkpoint_path}")
        else:
            # Create trainer and train
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                config=config,
                device=device
            )
            
            # Train model
            training_results = trainer.train()
            
            # Plot training history
            plot_training_history(
                training_results['history'],
                save_path=f"assets/{config.model.model_type}_training_history.png"
            )
            
            print(f"\\nTraining completed!")
            print(f"Best validation accuracy: {training_results['best_val_score']:.4f}")
            print(f"Test metrics: {training_results['test_metrics']}")


if __name__ == "__main__":
    main()
