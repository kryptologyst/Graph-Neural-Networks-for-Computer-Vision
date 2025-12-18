"""Evaluation metrics and utilities for GNN models."""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report
)
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(
    y_true: List[int], 
    y_pred: List[int], 
    metrics: List[str],
    y_proba: Optional[List[List[float]]] = None
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        metrics: List of metrics to compute
        y_proba: Predicted probabilities (for ROC-AUC)
        
    Returns:
        Dict of metric names and values
    """
    results = {}
    
    if 'accuracy' in metrics:
        results['accuracy'] = accuracy_score(y_true, y_pred)
    
    if 'f1_macro' in metrics or 'f1_micro' in metrics:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        if 'f1_macro' in metrics:
            results['f1_macro'] = np.mean(f1)
        if 'f1_micro' in metrics:
            precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
                y_true, y_pred, average='micro'
            )
            results['f1_micro'] = f1_micro
    
    if 'auroc' in metrics and y_proba is not None:
        try:
            # Handle multi-class case
            if len(np.unique(y_true)) > 2:
                results['auroc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            else:
                results['auroc'] = roc_auc_score(y_true, y_proba[:, 1])
        except ValueError:
            results['auroc'] = 0.0
    
    return results


def plot_confusion_matrix(
    y_true: List[int], 
    y_pred: List[int], 
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None) -> None:
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    if 'val_metrics' in history and history['val_metrics']:
        val_acc = [metrics.get('accuracy', 0) for metrics in history['val_metrics']]
        axes[1].plot(val_acc, label='Validation Accuracy')
        axes[1].set_title('Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_model_leaderboard(results: Dict[str, Dict[str, float]]) -> None:
    """
    Create a model comparison leaderboard.
    
    Args:
        results: Dictionary with model names as keys and metrics as values
    """
    import pandas as pd
    
    # Convert to DataFrame
    df = pd.DataFrame(results).T
    
    # Sort by accuracy (or primary metric)
    primary_metric = 'accuracy' if 'accuracy' in df.columns else df.columns[0]
    df = df.sort_values(primary_metric, ascending=False)
    
    print("\\n" + "="*60)
    print("MODEL LEADERBOARD")
    print("="*60)
    print(df.round(4))
    print("="*60)
    
    # Save to CSV
    df.to_csv("assets/model_leaderboard.csv")
    print("Leaderboard saved to assets/model_leaderboard.csv")


def analyze_predictions(
    y_true: List[int], 
    y_pred: List[int], 
    y_proba: Optional[List[List[float]]] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Analyze prediction results.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        class_names: Names of classes
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {}
    
    # Basic statistics
    analysis['total_samples'] = len(y_true)
    analysis['correct_predictions'] = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    analysis['accuracy'] = analysis['correct_predictions'] / analysis['total_samples']
    
    # Per-class analysis
    unique_classes = np.unique(y_true)
    class_analysis = {}
    
    for cls in unique_classes:
        cls_mask = np.array(y_true) == cls
        cls_pred = np.array(y_pred)[cls_mask]
        cls_correct = sum(cls_pred == cls)
        cls_total = sum(cls_mask)
        
        class_analysis[f'class_{cls}'] = {
            'total_samples': cls_total,
            'correct_predictions': cls_correct,
            'accuracy': cls_correct / cls_total if cls_total > 0 else 0
        }
    
    analysis['per_class'] = class_analysis
    
    # Confidence analysis (if probabilities available)
    if y_proba is not None:
        confidences = [max(probs) for probs in y_proba]
        analysis['avg_confidence'] = np.mean(confidences)
        analysis['min_confidence'] = np.min(confidences)
        analysis['max_confidence'] = np.max(confidences)
    
    return analysis


def visualize_attention_weights(
    attention_weights: List[torch.Tensor],
    edge_index: torch.Tensor,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize attention weights from GAT model.
    
    Args:
        attention_weights: List of attention weight tensors
        edge_index: Edge indices
        save_path: Path to save the plot
    """
    if not attention_weights:
        print("No attention weights to visualize")
        return
    
    # Use the last layer's attention weights
    att_weights = attention_weights[-1]
    
    # Average across heads if multiple heads
    if att_weights.dim() > 1:
        att_weights = att_weights.mean(dim=1)
    
    # Create attention matrix
    num_nodes = edge_index.max().item() + 1
    att_matrix = torch.zeros(num_nodes, num_nodes)
    
    for i, (src, dst) in enumerate(edge_index.t()):
        att_matrix[src, dst] = att_weights[i]
    
    # Plot attention matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(att_matrix.numpy(), cmap='viridis', cbar=True)
    plt.title('Attention Weights Heatmap')
    plt.xlabel('Target Node')
    plt.ylabel('Source Node')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
