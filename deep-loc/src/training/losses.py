"""
Loss functions for regression and classification tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionLoss(nn.Module):
    """
    Loss function for 3D position regression.
    Supports MSE, Smooth L1, and Huber loss.
    """

    def __init__(self, loss_type='mse', reduction='mean'):
        """
        Initialize regression loss.

        Args:
            loss_type: 'mse', 'smooth_l1', or 'huber'
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()

        self.loss_type = loss_type
        self.reduction = reduction

        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction=reduction)
        elif loss_type == 'smooth_l1':
            self.loss_fn = nn.SmoothL1Loss(reduction=reduction)
        elif loss_type == 'huber':
            self.loss_fn = nn.HuberLoss(reduction=reduction)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, pred, target):
        """
        Compute loss.

        Args:
            pred: (batch, 3) predicted positions
            target: (batch, 3) ground truth positions

        Returns:
            scalar loss value
        """
        return self.loss_fn(pred, target)


class ClassificationLoss(nn.Module):
    """
    Loss function for classification task.
    """

    def __init__(self, label_smoothing=0.0, reduction='mean'):
        """
        Initialize classification loss.

        Args:
            label_smoothing: label smoothing factor (0.0 to 1.0)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()

        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, labels):
        """
        Compute cross-entropy loss.

        Args:
            logits: (batch, num_classes) predicted logits
            labels: (batch,) ground truth labels

        Returns:
            scalar loss value
        """
        return F.cross_entropy(
            logits, labels,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction
        )


def build_loss_fn(config):
    """
    Build loss function based on config.

    Args:
        config: configuration object

    Returns:
        loss function
    """
    if config.task == 'regression':
        return RegressionLoss(
            loss_type=config.regression_loss,
            reduction='mean'
        )
    elif config.task == 'classification':
        return ClassificationLoss(
            label_smoothing=config.label_smoothing,
            reduction='mean'
        )
    else:
        raise ValueError(f"Unknown task: {config.task}")


if __name__ == '__main__':
    # Test regression loss
    print("Testing Regression Loss:")
    reg_loss = RegressionLoss(loss_type='mse')
    pred = torch.randn(8, 3)
    target = torch.randn(8, 3)
    loss = reg_loss(pred, target)
    print(f"  Pred shape: {pred.shape}")
    print(f"  Target shape: {target.shape}")
    print(f"  Loss: {loss.item():.4f}")
    assert loss.dim() == 0, "Loss should be scalar!"
    print("  Test passed!")

    # Test classification loss
    print("\nTesting Classification Loss:")
    cls_loss = ClassificationLoss(label_smoothing=0.1)
    logits = torch.randn(8, 100)
    labels = torch.randint(0, 100, (8,))
    loss = cls_loss(logits, labels)
    print(f"  Logits shape: {logits.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Loss: {loss.item():.4f}")
    assert loss.dim() == 0, "Loss should be scalar!"
    print("  Test passed!")
