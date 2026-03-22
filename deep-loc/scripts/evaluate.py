"""
Evaluation script for trained models.
"""

import os
import sys
import argparse
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.task_configs import get_config
from src.data.dataset import load_data, create_dataloaders
from src.models.localization_model import IndoorLocalizationModel
from src.training.metrics import compute_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate indoor localization model')

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default='data.mat',
                       help='Path to data.mat file')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')

    return parser.parse_args()


def evaluate(model, data_loader, device, task='regression'):
    """
    Evaluate model on dataset.

    Args:
        model: trained model
        data_loader: data loader
        device: device to use
        task: 'regression' or 'classification'

    Returns:
        dict of metrics
    """
    model.eval()
    all_preds = []
    all_targets = []

    print("Evaluating...")
    with torch.no_grad():
        for data, target in data_loader:
            # Move data to device
            data = data.to(device)
            target = target.to(device)

            # Forward pass
            output = model(data)

            # Store predictions and targets
            all_preds.append(output.cpu())
            all_targets.append(target.cpu())

    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute metrics
    metrics = compute_metrics(all_preds, all_targets, task=task)

    return metrics, all_preds, all_targets


def main():
    """Main evaluation function."""
    args = parse_args()

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # Get config from checkpoint
    config_dict = checkpoint.get('config', {})
    task = config_dict.get('task', 'regression')

    # Create config
    config = get_config(task)

    # Update config with checkpoint values
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)

    config.data_path = args.data_path
    config.batch_size = args.batch_size

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    train_dataset, val_dataset = load_data(
        config.data_path,
        task=config.task,
        normalize_positions=config.normalize_positions,
        pos_min=config.pos_min,
        pos_max=config.pos_max
    )

    _, val_loader = create_dataloaders(
        train_dataset, val_dataset,
        batch_size=config.batch_size,
        num_workers=0,  # Use 0 for evaluation to avoid multiprocessing issues
        pin_memory=False
    )

    # Create model
    print("Creating model...")
    model = IndoorLocalizationModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Evaluate
    metrics, preds, targets = evaluate(model, val_loader, device, task=config.task)

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Dataset: Validation")
    print(f"Number of samples: {len(val_dataset)}")
    print(f"Task: {config.task}")
    print("-" * 80)

    for key, value in metrics.items():
        print(f"{key:20s}: {value:.6f}")

    print("=" * 80)

    # Denormalize predictions and targets for regression
    if config.task == 'regression' and config.normalize_positions:
        preds_denorm = val_dataset.denormalize_position(preds.numpy())
        targets_denorm = val_dataset.denormalize_position(targets.numpy())

        print("\nDenormalized Metrics (in meters):")
        print("-" * 80)

        # Recompute metrics on denormalized data
        preds_denorm_tensor = torch.from_numpy(preds_denorm).float()
        targets_denorm_tensor = torch.from_numpy(targets_denorm).float()
        denorm_metrics = compute_metrics(preds_denorm_tensor, targets_denorm_tensor, task='regression')

        for key, value in denorm_metrics.items():
            print(f"{key:20s}: {value:.6f}")

        print("=" * 80)


if __name__ == '__main__':
    main()
