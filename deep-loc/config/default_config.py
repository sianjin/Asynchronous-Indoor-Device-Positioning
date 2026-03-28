"""
Default configuration for indoor localization model.
"""

import os


class DefaultConfig:
    """Default hyperparameters and settings."""

    # Data paths
    data_path = 'data.mat'
    save_dir = './checkpoints'
    log_dir = './logs'

    # Data parameters
    input_shape = (48, 32)  # (Ntap, 2M)
    num_anchors = 4
    num_workers = 4
    pin_memory = True

    # Position bounds (from data analysis)
    pos_min = [0.25, 0.25, 0.80]  # [x_min, y_min, z_min]
    pos_max = [4.75, 7.75, 1.80]  # [x_max, y_max, z_max]

    # Model architecture - Encoder (Deep CNN - conservative upgrade)
    # 4-layer CNN with 256 filters each, using stride-based downsampling
    # Conservative upgrade from baseline: same architecture, just deeper/wider
    embed_dim = 256
    cnn_channels = [256, 256, 256, 256]  # 4 layers, all 256 filters
    # mlp_hidden_dim = 256  # Removed: no longer using MLP branch
    encoder_dropout = 0.2  # Increased from 0.1 to match MATLAB

    # Model architecture - Transformer
    num_heads = 4
    num_layers = 2
    ff_dim = 512
    transformer_dropout = 0.1

    # Model architecture - Task heads
    regression_hidden_dims = [128, 64]
    classification_hidden_dims = [128, 64]
    head_dropout = 0.1
    num_classes = 100  # To be updated based on actual classification labels

    # Training parameters
    task = 'regression'  # 'regression' or 'classification'
    batch_size = 32
    num_epochs = 200
    learning_rate = 1e-3
    weight_decay = 1e-4

    # Optimizer
    optimizer = 'adamw'  # 'adam', 'adamw', 'sgd'

    # Learning rate scheduler
    lr_scheduler = 'cosine'  # 'cosine', 'step', 'plateau', 'none'
    warmup_epochs = 10
    min_lr = 1e-6
    step_size = 30  # For step scheduler
    gamma = 0.1  # For step scheduler
    patience_scheduler = 10  # For plateau scheduler

    # Loss parameters
    regression_loss = 'mse'  # 'mse', 'smooth_l1', 'huber'
    normalize_positions = True  # Normalize positions to [0, 1] for training
    classification_loss = 'cross_entropy'
    label_smoothing = 0.0

    # Early stopping
    early_stopping = True
    patience = 30
    min_delta = 1e-4

    # Logging
    log_interval = 10  # Log every N batches
    experiment_name = 'baseline'

    # Device
    device = 'cuda'  # Will be set to 'cpu' if CUDA is not available

    # Random seed
    seed = 42

    def __init__(self, **kwargs):
        """Allow overriding defaults via kwargs."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")

    def to_dict(self):
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def __repr__(self):
        """String representation of config."""
        config_str = "Configuration:\n"
        for key, value in sorted(self.to_dict().items()):
            config_str += f"  {key}: {value}\n"
        return config_str
