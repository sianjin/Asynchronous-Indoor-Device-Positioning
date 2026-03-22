"""
Task-specific output heads for regression and classification.
"""

import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    """
    MLP head for 3D position regression.

    Input: (batch, input_dim)
    Output: (batch, 3) - (x, y, z) coordinates
    """

    def __init__(self, input_dim=256, hidden_dims=[128, 64], dropout=0.1):
        """
        Initialize regression head.

        Args:
            input_dim: input feature dimension
            hidden_dims: list of hidden layer dimensions
            dropout: dropout probability
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer (no activation for regression)
        layers.append(nn.Linear(prev_dim, 3))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch, input_dim) tensor

        Returns:
            (batch, 3) tensor of (x, y, z) positions
        """
        return self.mlp(x)


class ClassificationHead(nn.Module):
    """
    MLP head for classification.

    Input: (batch, input_dim)
    Output: (batch, num_classes) - logits
    """

    def __init__(self, input_dim=256, hidden_dims=[128, 64], num_classes=100, dropout=0.1):
        """
        Initialize classification head.

        Args:
            input_dim: input feature dimension
            hidden_dims: list of hidden layer dimensions
            num_classes: number of output classes
            dropout: dropout probability
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer (no softmax - use logits for CrossEntropyLoss)
        layers.append(nn.Linear(prev_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch, input_dim) tensor

        Returns:
            (batch, num_classes) tensor of logits
        """
        return self.mlp(x)


if __name__ == '__main__':
    # Test regression head
    reg_head = RegressionHead(input_dim=256, hidden_dims=[128, 64])
    x = torch.randn(8, 256)
    out = reg_head(x)
    print(f"Regression Head:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Expected: (8, 3)")
    assert out.shape == (8, 3), "Regression head output shape mismatch!"
    print("  Test passed!")

    # Test classification head
    cls_head = ClassificationHead(input_dim=256, hidden_dims=[128, 64], num_classes=100)
    out = cls_head(x)
    print(f"\nClassification Head:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Expected: (8, 100)")
    assert out.shape == (8, 100), "Classification head output shape mismatch!"
    print("  Test passed!")
