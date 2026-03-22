"""
Main training script for indoor localization.
"""

import os
import sys
import argparse
import random
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.task_configs import get_config
from src.data.dataset import load_data, create_dataloaders
from src.models.localization_model import IndoorLocalizationModel
from src.training.trainer import Trainer
from src.utils.logger import Logger


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train indoor localization model')

    # Task and config
    parser.add_argument('--task', type=str, default='regression',
                       choices=['regression', 'classification'],
                       help='Task type (regression or classification)')
    parser.add_argument('--config', type=str, default=None,
                       help='Config name (regression or classification)')

    # Data
    parser.add_argument('--data_path', type=str, default='data.mat',
                       help='Path to data.mat file')

    # Training
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=None,
                       help='Weight decay')

    # Model
    parser.add_argument('--embed_dim', type=int, default=None,
                       help='Embedding dimension')
    parser.add_argument('--num_layers', type=int, default=None,
                       help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=None,
                       help='Number of attention heads')

    # Misc
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name for logging')

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Get configuration
    config_name = args.config if args.config is not None else args.task
    config = get_config(config_name)

    # Override config with command line arguments
    if args.data_path is not None:
        config.data_path = args.data_path
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.embed_dim is not None:
        config.embed_dim = args.embed_dim
    if args.num_layers is not None:
        config.num_layers = args.num_layers
    if args.num_heads is not None:
        config.num_heads = args.num_heads
    if args.seed is not None:
        config.seed = args.seed
    if args.device is not None:
        config.device = args.device
    if args.experiment_name is not None:
        config.experiment_name = args.experiment_name

    # Set random seed
    set_seed(config.seed)

    # Create logger
    logger = Logger(config.log_dir, config.experiment_name)
    logger.log_config(config)

    # Load data
    logger.log("Loading data...")
    train_dataset, val_dataset = load_data(
        config.data_path,
        task=config.task,
        normalize_positions=config.normalize_positions,
        pos_min=config.pos_min,
        pos_max=config.pos_max
    )

    logger.log(f"Train dataset size: {len(train_dataset)}")
    logger.log(f"Val dataset size: {len(val_dataset)}")

    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    # Create model
    logger.log("Creating model...")
    model = IndoorLocalizationModel(config)

    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config, logger)

    # Resume from checkpoint if specified
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()

    logger.log("Training completed successfully!")


if __name__ == '__main__':
    main()
