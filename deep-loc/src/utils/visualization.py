"""
Visualization utilities for indoor localization.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


def plot_training_curves(metrics_history, save_path=None):
    """
    Plot training and validation loss curves.

    Args:
        metrics_history: dict with 'epochs', 'train_loss', 'val_loss' keys
        save_path: path to save plot (optional)
    """
    epochs = metrics_history['epochs']
    train_loss = metrics_history['train_loss']
    val_loss = metrics_history['val_loss']

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training Loss', linewidth=2)
    plt.plot(epochs, val_loss, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_metrics(metrics_history, metric_name, save_path=None):
    """
    Plot a specific metric over epochs.

    Args:
        metrics_history: dict with metrics
        metric_name: name of metric to plot
        save_path: path to save plot (optional)
    """
    if metric_name not in metrics_history:
        print(f"Metric '{metric_name}' not found in history")
        return

    epochs = metrics_history['epochs']
    values = metrics_history[metric_name]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, values, linewidth=2, color='#2E86AB')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
    plt.title(f'{metric_name.replace("_", " ").title()} Over Training', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_predictions_vs_ground_truth(predictions, ground_truth, save_path=None):
    """
    Plot predicted vs ground truth positions for regression task.

    Args:
        predictions: (N, 3) array of predicted positions
        ground_truth: (N, 3) array of ground truth positions
        save_path: path to save plot (optional)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels = ['X', 'Y', 'Z']

    for idx, (ax, label) in enumerate(zip(axes, labels)):
        pred = predictions[:, idx]
        true = ground_truth[:, idx]

        # Scatter plot
        ax.scatter(true, pred, alpha=0.6, s=20)

        # Perfect prediction line
        min_val = min(true.min(), pred.min())
        max_val = max(true.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        ax.set_xlabel(f'Ground Truth {label} (m)', fontsize=11)
        ax.set_ylabel(f'Predicted {label} (m)', fontsize=11)
        ax.set_title(f'{label}-axis Predictions', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add correlation coefficient
        corr = np.corrcoef(true, pred)[0, 1]
        ax.text(0.05, 0.95, f'Corr: {corr:.3f}', transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_error_distribution(predictions, ground_truth, save_path=None):
    """
    Plot distribution of prediction errors.

    Args:
        predictions: (N, 3) array of predicted positions
        ground_truth: (N, 3) array of ground truth positions
        save_path: path to save plot (optional)
    """
    # Compute 3D Euclidean distances
    errors = np.sqrt(((predictions - ground_truth) ** 2).sum(axis=1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(errors, bins=50, alpha=0.7, color='#2E86AB', edgecolor='black')
    axes[0].axvline(errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.3f}m')
    axes[0].axvline(np.median(errors), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.3f}m')
    axes[0].set_xlabel('Error (m)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Error Distribution', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Cumulative distribution
    sorted_errors = np.sort(errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100

    axes[1].plot(sorted_errors, cumulative, linewidth=2, color='#2E86AB')
    axes[1].set_xlabel('Error (m)', fontsize=12)
    axes[1].set_ylabel('Cumulative Percentage (%)', fontsize=12)
    axes[1].set_title('Cumulative Error Distribution', fontsize=13)
    axes[1].grid(True, alpha=0.3)

    # Add percentile markers
    for percentile in [50, 90, 95]:
        error_val = np.percentile(errors, percentile)
        axes[1].axvline(error_val, linestyle='--', alpha=0.7, label=f'{percentile}th: {error_val:.3f}m')

    axes[1].legend(fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_3d_positions(predictions, ground_truth, save_path=None):
    """
    3D scatter plot of predicted and ground truth positions.

    Args:
        predictions: (N, 3) array of predicted positions
        ground_truth: (N, 3) array of ground truth positions
        save_path: path to save plot (optional)
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot ground truth
    ax.scatter(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2],
               c='blue', marker='o', s=50, alpha=0.6, label='Ground Truth')

    # Plot predictions
    ax.scatter(predictions[:, 0], predictions[:, 1], predictions[:, 2],
               c='red', marker='^', s=50, alpha=0.6, label='Predictions')

    # Draw connecting lines
    for i in range(len(predictions)):
        ax.plot([ground_truth[i, 0], predictions[i, 0]],
                [ground_truth[i, 1], predictions[i, 1]],
                [ground_truth[i, 2], predictions[i, 2]],
                'k-', alpha=0.2, linewidth=0.5)

    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_zlabel('Z (m)', fontsize=11)
    ax.set_title('3D Position Predictions', fontsize=13)
    ax.legend(fontsize=10)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_per_axis_errors(predictions, ground_truth, save_path=None):
    """
    Box plot of per-axis errors.

    Args:
        predictions: (N, 3) array of predicted positions
        ground_truth: (N, 3) array of ground truth positions
        save_path: path to save plot (optional)
    """
    errors = predictions - ground_truth

    fig, ax = plt.subplots(figsize=(10, 6))

    bp = ax.boxplot([errors[:, 0], errors[:, 1], errors[:, 2]],
                     labels=['X', 'Y', 'Z'],
                     patch_artist=True,
                     notch=True)

    # Customize colors
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_ylabel('Error (m)', fontsize=12)
    ax.set_title('Per-Axis Error Distribution', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def compute_confusion_matrix(ground_truth, predictions, num_classes):
    """
    Compute confusion matrix without sklearn.

    Args:
        ground_truth: numpy array of true labels
        predictions: numpy array of predicted labels
        num_classes: number of classes

    Returns:
        confusion matrix of shape (num_classes, num_classes)
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(ground_truth, predictions):
        cm[true_label, pred_label] += 1
    return cm


def plot_confusion_matrix(predictions, ground_truth, category_names, checkpoint_path=None, output_dir='results/localization', save_path=None):
    """
    Plot and save confusion matrix for classification results.

    Args:
        predictions: numpy array of shape (samples,) - predicted class indices (0-indexed)
        ground_truth: numpy array of shape (samples,) - ground truth class indices (0-indexed)
        category_names: list of category names
        checkpoint_path: path to the checkpoint file (used for naming, optional)
        output_dir: directory to save the plot (used if checkpoint_path is provided)
        save_path: direct path to save plot (optional, overrides output_dir)

    Returns:
        path to saved plot file
    """
    # Compute confusion matrix
    cm = compute_confusion_matrix(ground_truth, predictions, len(category_names))

    # Determine save path
    if save_path:
        plot_filepath = Path(save_path)
    elif checkpoint_path:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename
        checkpoint_name = Path(checkpoint_path).stem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_filename = f'confusion_matrix_{checkpoint_name}_{timestamp}.png'
        plot_filepath = output_path / plot_filename
    else:
        # Default save path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_filepath = output_path / f'confusion_matrix_{timestamp}.png'

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot confusion matrix as heatmap using imshow
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', rotation=270, labelpad=20)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(category_names)))
    ax.set_yticks(np.arange(len(category_names)))
    ax.set_xticklabels(category_names, rotation=45, ha='right')
    ax.set_yticklabels(category_names, rotation=0)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=10)

    # Set title
    if checkpoint_path:
        checkpoint_name = Path(checkpoint_path).stem
        ax.set_title(f'Confusion Matrix - {checkpoint_name}', fontsize=14, fontweight='bold', pad=20)
    else:
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)

    ax.set_xlabel('Predicted Category', fontsize=12)
    ax.set_ylabel('True Category', fontsize=12)
    plt.tight_layout()

    # Save the figure
    plt.savefig(plot_filepath, dpi=150, bbox_inches='tight')
    plt.close()

    # Print save information
    print(f"\n{'=' * 80}")
    print("CONFUSION MATRIX SAVED")
    print("=" * 80)
    print(f"File: {plot_filepath}")
    print(f"Format: PNG image (150 DPI)")
    print(f"Size: {cm.shape[0]}×{cm.shape[1]} confusion matrix")
    print("=" * 80)

    # Print accuracy per class
    print("\nPer-Class Accuracy:")
    print("-" * 80)
    for i, cat in enumerate(category_names):
        if cm[i].sum() > 0:
            accuracy = cm[i, i] / cm[i].sum() * 100
            print(f"  {cat:20s}: {accuracy:6.2f}% ({cm[i, i]:3d}/{cm[i].sum():3d})")
    print("=" * 80)

    return str(plot_filepath)


if __name__ == '__main__':
    # Test visualization functions
    print("Testing visualization utilities...")

    # Create dummy data
    n_samples = 100
    predictions = np.random.randn(n_samples, 3) * 0.5 + np.array([2.5, 4.0, 1.3])
    ground_truth = predictions + np.random.randn(n_samples, 3) * 0.3

    # Test plots
    print("Generating test plots...")

    plot_predictions_vs_ground_truth(predictions, ground_truth)
    print("✓ Predictions vs Ground Truth plot")

    plot_error_distribution(predictions, ground_truth)
    print("✓ Error Distribution plot")

    plot_per_axis_errors(predictions, ground_truth)
    print("✓ Per-Axis Errors plot")

    print("\nAll visualization tests passed!")
