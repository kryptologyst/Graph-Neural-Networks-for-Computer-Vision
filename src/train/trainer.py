"""Training utilities for GNN models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from tqdm import tqdm
import logging

from ..utils.device import get_device, save_checkpoint, load_checkpoint
from ..eval.metrics import compute_metrics


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop early.
        
        Args:
            score: Current validation score
            model: Model to potentially restore weights
            
        Returns:
            bool: True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        
        return False
    
    def save_checkpoint(self, model: nn.Module):
        """Save model checkpoint."""
        self.best_weights = model.state_dict().copy()


class Trainer:
    """Main trainer class for GNN models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        config: Any,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device or get_device(config.system.device)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Setup scheduler
        if config.training.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.training.epochs)
        elif config.training.scheduler == "step":
            self.scheduler = StepLR(self.optimizer, step_size=config.training.epochs // 3, gamma=0.1)
        elif config.training.scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', patience=5, factor=0.5)
        else:
            self.scheduler = None
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup early stopping
        if config.training.early_stopping:
            self.early_stopping = EarlyStopping(patience=config.training.patience)
        else:
            self.early_stopping = None
        
        # Mixed precision
        if config.training.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, config.system.log_level))
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    out = self.model(batch.x, batch.edge_index, batch.batch)
                    loss = self.criterion(out, batch.y)
                
                self.scaler.scale(loss).backward()
                
                if self.config.training.gradient_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = self.criterion(out, batch.y)
                loss.backward()
                
                if self.config.training.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip)
                
                self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                
                out = self.model(batch.x, batch.edge_index, batch.batch)
                loss = self.criterion(out, batch.y)
                
                total_loss += loss.item()
                
                preds = torch.argmax(out, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        metrics = compute_metrics(all_labels, all_preds, self.config.evaluation.metrics)
        
        return avg_loss, metrics
    
    def test(self) -> Dict[str, float]:
        """Test the model."""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(self.device)
                
                out = self.model(batch.x, batch.edge_index, batch.batch)
                probs = torch.softmax(out, dim=1)
                
                preds = torch.argmax(out, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        metrics = compute_metrics(all_labels, all_preds, self.config.evaluation.metrics)
        
        # Save predictions if requested
        if self.config.evaluation.save_predictions:
            import os
            os.makedirs("assets", exist_ok=True)
            np.savez("assets/test_predictions.npz", 
                    predictions=np.array(all_preds),
                    labels=np.array(all_labels),
                    probabilities=np.array(all_probs))
        
        return metrics
    
    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        self.logger.info(f"Starting training for {self.config.training.epochs} epochs")
        self.logger.info(f"Model: {self.config.model.model_type}")
        self.logger.info(f"Device: {self.device}")
        
        best_val_score = 0.0
        best_epoch = 0
        
        for epoch in range(self.config.training.epochs):
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            val_loss, val_metrics = self.validate()
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('accuracy', 0))
                else:
                    self.scheduler.step()
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch+1:3d}/{self.config.training.epochs}: "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_metrics.get('accuracy', 0):.4f}"
            )
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)
            
            # Check for best model
            val_score = val_metrics.get('accuracy', 0)
            if val_score > best_val_score:
                best_val_score = val_score
                best_epoch = epoch
                
                # Save best model
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss, val_metrics,
                    f"checkpoints/best_model_{self.config.model.model_type}.pt"
                )
            
            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping(val_score, self.model):
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        self.logger.info(f"Training completed. Best validation accuracy: {best_val_score:.4f} at epoch {best_epoch+1}")
        
        # Load best model for testing
        load_checkpoint(self.model, None, f"checkpoints/best_model_{self.config.model.model_type}.pt")
        
        # Final test
        test_metrics = self.test()
        self.logger.info(f"Final test metrics: {test_metrics}")
        
        return {
            'best_val_score': best_val_score,
            'best_epoch': best_epoch,
            'test_metrics': test_metrics,
            'history': self.history
        }
