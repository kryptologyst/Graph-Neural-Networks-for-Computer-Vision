"""Configuration management for Graph Neural Networks for Computer Vision."""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import yaml


@dataclass
class DataConfig:
    """Data configuration."""
    dataset_name: str = "CIFAR10"
    data_root: str = "data"
    batch_size: int = 32
    num_workers: int = 4
    train_size: int = 500
    test_size: int = 100
    val_size: int = 100
    superpixel_segments: int = 75
    superpixel_compactness: float = 10.0
    augment: bool = True
    normalize: bool = True


@dataclass
class ModelConfig:
    """Model configuration."""
    model_type: str = "gcn"  # gcn, graphsage, gat, gin, transformer
    hidden_channels: int = 64
    num_layers: int = 2
    dropout: float = 0.5
    activation: str = "relu"
    use_batch_norm: bool = True
    use_residual: bool = False
    pooling: str = "mean"  # mean, max, attention, set2set


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 50
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    scheduler: str = "cosine"  # cosine, step, plateau
    patience: int = 10
    early_stopping: bool = True
    gradient_clip: Optional[float] = None
    mixed_precision: bool = False


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1_macro", "f1_micro", "auroc"])
    save_predictions: bool = True
    save_embeddings: bool = True
    visualize_attention: bool = True
    explain_predictions: bool = True


@dataclass
class SystemConfig:
    """System configuration."""
    device: str = "auto"  # auto, cuda, mps, cpu
    seed: int = 42
    deterministic: bool = True
    num_threads: Optional[int] = None
    log_level: str = "INFO"


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False, indent=2)
    
    def update(self, **kwargs) -> None:
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # Handle nested updates
                for section in [self.data, self.model, self.training, self.evaluation, self.system]:
                    if hasattr(section, key):
                        setattr(section, key, value)
                        break
