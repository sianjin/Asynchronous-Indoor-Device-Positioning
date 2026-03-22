"""
Evaluation metrics for regression and classification tasks.
"""

import torch
import numpy as np


class LocalizationMetrics:
    """Metrics for regression task (3D position estimation)."""

    @staticmethod
    def euclidean_distance(pred, target):
        """
        Compute 3D Euclidean distance in meters.

        Args:
            pred: (N, 3) predicted positions
            target: (N, 3) ground truth positions

        Returns:
            (N,) distances
        """
        return torch.sqrt(((pred - target) ** 2).sum(dim=-1))

    @staticmethod
    def mean_distance_error(pred, target):
        """Mean 3D distance error."""
        return LocalizationMetrics.euclidean_distance(pred, target).mean().item()

    @staticmethod
    def median_distance_error(pred, target):
        """Median 3D distance error."""
        return LocalizationMetrics.euclidean_distance(pred, target).median().item()

    @staticmethod
    def percentile_error(pred, target, percentile=90):
        """Nth percentile error."""
        distances = LocalizationMetrics.euclidean_distance(pred, target)
        return torch.quantile(distances, percentile / 100).item()

    @staticmethod
    def per_axis_mae(pred, target):
        """
        Mean Absolute Error for each axis (x, y, z).

        Returns:
            dict with 'x', 'y', 'z' keys
        """
        mae = (pred - target).abs().mean(dim=0)
        return {
            'x': mae[0].item(),
            'y': mae[1].item(),
            'z': mae[2].item()
        }

    @staticmethod
    def per_axis_rmse(pred, target):
        """
        Root Mean Squared Error for each axis (x, y, z).

        Returns:
            dict with 'x', 'y', 'z' keys
        """
        mse = ((pred - target) ** 2).mean(dim=0)
        rmse = torch.sqrt(mse)
        return {
            'x': rmse[0].item(),
            'y': rmse[1].item(),
            'z': rmse[2].item()
        }

    @staticmethod
    def compute_all(pred, target):
        """
        Compute all regression metrics.

        Returns:
            dict of all metrics
        """
        metrics = {
            'mean_error': LocalizationMetrics.mean_distance_error(pred, target),
            'median_error': LocalizationMetrics.median_distance_error(pred, target),
            '90th_percentile': LocalizationMetrics.percentile_error(pred, target, 90),
            '95th_percentile': LocalizationMetrics.percentile_error(pred, target, 95),
        }

        # Add per-axis MAE
        mae_dict = LocalizationMetrics.per_axis_mae(pred, target)
        metrics.update({f'mae_{k}': v for k, v in mae_dict.items()})

        # Add per-axis RMSE
        rmse_dict = LocalizationMetrics.per_axis_rmse(pred, target)
        metrics.update({f'rmse_{k}': v for k, v in rmse_dict.items()})

        return metrics


class ClassificationMetrics:
    """Metrics for classification task."""

    @staticmethod
    def accuracy(logits, labels):
        """
        Compute classification accuracy.

        Args:
            logits: (N, num_classes) predicted logits
            labels: (N,) ground truth labels

        Returns:
            accuracy (0-1)
        """
        pred = logits.argmax(dim=-1)
        return (pred == labels).float().mean().item()

    @staticmethod
    def top_k_accuracy(logits, labels, k=5):
        """
        Compute top-k accuracy.

        Args:
            logits: (N, num_classes) predicted logits
            labels: (N,) ground truth labels
            k: top-k

        Returns:
            top-k accuracy (0-1)
        """
        _, pred = logits.topk(k, dim=-1)
        correct = (pred == labels.unsqueeze(-1)).any(dim=-1)
        return correct.float().mean().item()

    @staticmethod
    def compute_all(logits, labels):
        """
        Compute all classification metrics.

        Returns:
            dict of all metrics
        """
        metrics = {
            'accuracy': ClassificationMetrics.accuracy(logits, labels),
            'top5_accuracy': ClassificationMetrics.top_k_accuracy(logits, labels, k=5),
        }
        return metrics


def compute_metrics(pred, target, task='regression'):
    """
    Compute metrics based on task.

    Args:
        pred: predictions
        target: ground truth
        task: 'regression' or 'classification'

    Returns:
        dict of metrics
    """
    if task == 'regression':
        return LocalizationMetrics.compute_all(pred, target)
    elif task == 'classification':
        return ClassificationMetrics.compute_all(pred, target)
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == '__main__':
    # Test regression metrics
    print("Testing Regression Metrics:")
    pred = torch.tensor([[1.0, 2.0, 1.0], [3.0, 4.0, 1.5], [2.0, 3.0, 1.2]])
    target = torch.tensor([[1.1, 2.1, 1.05], [2.8, 3.9, 1.45], [2.2, 3.2, 1.25]])

    metrics = LocalizationMetrics.compute_all(pred, target)
    print("  Metrics:")
    for key, value in metrics.items():
        print(f"    {key}: {value:.4f}")

    # Test classification metrics
    print("\nTesting Classification Metrics:")
    logits = torch.randn(100, 10)
    labels = torch.randint(0, 10, (100,))

    metrics = ClassificationMetrics.compute_all(logits, labels)
    print("  Metrics:")
    for key, value in metrics.items():
        print(f"    {key}: {value:.4f}")

    print("\nAll tests passed!")
