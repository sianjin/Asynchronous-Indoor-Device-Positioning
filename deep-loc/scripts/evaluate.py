"""
Evaluation script for trained models.
"""

import os
import sys
import argparse
import torch
import scipy.io
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.task_configs import get_config
from src.data.dataset import load_data, create_dataloaders
from src.models.localization_model import IndoorLocalizationModel
from src.training.metrics import compute_metrics
from src.utils.visualization import plot_confusion_matrix


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
    parser.add_argument('--save_results', action='store_true',
                       help='Save evaluation results to .mat file (regression or classification)')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Base directory to save results (default: results/positioning for regression, results/localization for classification)')

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

        # Save results to .mat file if requested
        if args.save_results:
            # Create positioning subfolder under base results directory
            output_dir = str(Path(args.output_dir) / 'positioning')
            save_positioning_results(
                predictions=preds_denorm,
                ground_truth=targets_denorm,
                checkpoint_path=args.checkpoint,
                output_dir=output_dir
            )
    elif config.task == 'regression' and not config.normalize_positions:
        # If positions are not normalized, save directly
        if args.save_results:
            # Create positioning subfolder under base results directory
            output_dir = str(Path(args.output_dir) / 'positioning')
            save_positioning_results(
                predictions=preds.numpy(),
                ground_truth=targets.numpy(),
                checkpoint_path=args.checkpoint,
                output_dir=output_dir
            )
    elif config.task == 'classification':
        # Convert predictions from logits to class indices
        pred_classes = torch.argmax(preds, dim=1).numpy()  # (samples,)
        target_classes = targets.numpy()  # (samples,)

        # Create localization subfolder under base results directory
        output_dir = str(Path(args.output_dir) / 'localization')

        # Define category names for plotting
        category_names = [
            'conference_room',
            'desk1',
            'desk2',
            'desk3',
            'desk4',
            'office',
            'storage'
        ]

        # Plot confusion matrix
        plot_confusion_matrix(
            predictions=pred_classes,
            ground_truth=target_classes,
            category_names=category_names,
            checkpoint_path=args.checkpoint,
            output_dir=output_dir
        )

        # Save classification results if requested
        if args.save_results:
            save_localization_results(
                predictions=pred_classes,
                ground_truth=target_classes,
                checkpoint_path=args.checkpoint,
                output_dir=output_dir
            )


def save_positioning_results(predictions, ground_truth, checkpoint_path, output_dir='results/positioning'):
    """
    Save regression positioning results to .mat file.

    Args:
        predictions: numpy array of shape (samples, 3) - predicted (x, y, z) positions
        ground_truth: numpy array of shape (samples, 3) - ground truth (x, y, z) positions
        checkpoint_path: path to the checkpoint file (used for naming)
        output_dir: directory to save results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename from checkpoint name and timestamp
    checkpoint_name = Path(checkpoint_path).stem  # e.g., 'regression_baseline_best'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mat_filename = f'positioning_{checkpoint_name}_{timestamp}.mat'
    mat_filepath = output_path / mat_filename

    # Prepare data for MATLAB
    # Shape: (samples, 3) where 3 = [x, y, z]
    mat_data = {
        'predicted_positions': predictions,  # (samples, 3)
        'ground_truth_positions': ground_truth,  # (samples, 3)
        'num_samples': predictions.shape[0],
        'checkpoint': checkpoint_name,
        'timestamp': timestamp
    }

    # Save to .mat file
    scipy.io.savemat(mat_filepath, mat_data)

    print(f"\n{'=' * 80}")
    print("POSITIONING RESULTS SAVED")
    print("=" * 80)
    print(f"File: {mat_filepath}")
    print(f"Format: MATLAB .mat file")
    print(f"Variables:")
    print(f"  - predicted_positions:    {predictions.shape} (samples × 3) [x, y, z]")
    print(f"  - ground_truth_positions: {ground_truth.shape} (samples × 3) [x, y, z]")
    print(f"  - num_samples:            {predictions.shape[0]}")
    print(f"  - checkpoint:             '{checkpoint_name}'")
    print(f"  - timestamp:              '{timestamp}'")
    print("=" * 80)
    print("\nLoad in MATLAB with:")
    print(f"  >> data = load('{mat_filepath}');")
    print(f"  >> predicted_pos = data.predicted_positions;  % {predictions.shape[0]}×3 matrix")
    print(f"  >> ground_truth_pos = data.ground_truth_positions;  % {predictions.shape[0]}×3 matrix")
    print("=" * 80)


def save_localization_results(predictions, ground_truth, checkpoint_path, output_dir='results/localization'):
    """
    Save classification localization results to .mat file.

    Args:
        predictions: numpy array of shape (samples,) - predicted class indices
        ground_truth: numpy array of shape (samples,) - ground truth class indices
        checkpoint_path: path to the checkpoint file (used for naming)
        output_dir: directory to save results
    """
    # Define category names (7 categories) - order matches MATLAB data
    category_names = [
        'conference_room',
        'desk1',
        'desk2',
        'desk3',
        'desk4',
        'office',
        'storage'
    ]

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename from checkpoint name and timestamp
    checkpoint_name = Path(checkpoint_path).stem  # e.g., 'classification_baseline_best'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mat_filename = f'localization_{checkpoint_name}_{timestamp}.mat'
    mat_filepath = output_path / mat_filename

    # Import numpy
    import numpy as np

    # Prepare indices (1-indexed for MATLAB, samples × 1)
    # Add 1 to convert from 0-indexed (Python) to 1-indexed (MATLAB)
    predicted_indices = (predictions + 1).reshape(-1, 1).astype(np.uint8)  # (samples, 1)
    ground_truth_indices = (ground_truth + 1).reshape(-1, 1).astype(np.uint8)  # (samples, 1)

    # Create category names array for MATLAB categorical
    category_names_array = np.array(category_names, dtype=object).reshape(-1, 1)

    # Prepare data for MATLAB
    # To create categorical arrays in MATLAB, we'll save:
    # 1. The indices (1-indexed for MATLAB)
    # 2. The category names
    # User can convert to categorical in MATLAB with: categorical(predicted_classes, 1:7, category_names)
    mat_data = {
        'predicted_classes': predicted_indices,  # (samples, 1) - 1-indexed for MATLAB
        'ground_truth_classes': ground_truth_indices,  # (samples, 1) - 1-indexed for MATLAB
        'category_names': category_names_array,  # (7, 1) cell array of category names
        'num_samples': len(predictions),
        'num_categories': len(category_names),
        'checkpoint': checkpoint_name,
        'timestamp': timestamp
    }

    # Save to .mat file
    scipy.io.savemat(mat_filepath, mat_data)

    print(f"\n{'=' * 80}")
    print("LOCALIZATION RESULTS SAVED")
    print("=" * 80)
    print(f"File: {mat_filepath}")
    print(f"Format: MATLAB .mat file")
    print(f"Variables:")
    print(f"  - predicted_classes:      {predicted_indices.shape} (samples × 1) [1-indexed class indices]")
    print(f"  - ground_truth_classes:   {ground_truth_indices.shape} (samples × 1) [1-indexed class indices]")
    print(f"  - category_names:         {len(category_names)} cell array [category names]")
    print(f"  - num_samples:            {len(predictions)}")
    print(f"  - num_categories:         {len(category_names)}")
    print(f"  - checkpoint:             '{checkpoint_name}'")
    print(f"  - timestamp:              '{timestamp}'")
    print("=" * 80)
    print("\nCategory names (1-indexed for MATLAB):")
    for i, cat in enumerate(category_names, 1):
        print(f"  {i}: {cat}")
    print("=" * 80)
    print("\nLoad in MATLAB and convert to categorical:")
    print(f"  >> data = load('{mat_filepath}');")
    print(f"  >> predicted_cat = categorical(data.predicted_classes, 1:{len(category_names)}, data.category_names);")
    print(f"  >> ground_truth_cat = categorical(data.ground_truth_classes, 1:{len(category_names)}, data.category_names);")
    print(f"  >> % predicted_cat and ground_truth_cat are now {len(predictions)}×1 categorical arrays")
    print("=" * 80)


if __name__ == '__main__':
    main()
