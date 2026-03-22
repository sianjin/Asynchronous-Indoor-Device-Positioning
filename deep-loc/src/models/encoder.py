"""
Hybrid CNN + MLP encoder for per-anchor feature extraction.
"""

import torch
import torch.nn as nn


class HybridEncoder(nn.Module):
    """
    Hybrid encoder combining CNN and MLP for per-anchor feature extraction.

    Architecture:
    - CNN branch: Extracts local spatial-temporal patterns via 2D convolutions
    - MLP branch: Captures global statistics via fully connected layers
    - Fusion: Concatenates features and projects to embedding dimension

    Input: (batch, Ntap, 2M) = (batch, 48, 32)
    Output: (batch, embed_dim)
    """

    def __init__(self, input_shape=(48, 32), embed_dim=256, cnn_channels=[32, 64, 128],
                 mlp_hidden_dim=256, dropout=0.1):
        """
        Initialize hybrid encoder.

        Args:
            input_shape: (Ntap, 2M) - input feature map shape
            embed_dim: output embedding dimension
            cnn_channels: list of channel dimensions for CNN layers
            mlp_hidden_dim: hidden dimension for MLP branch
            dropout: dropout probability
        """
        super().__init__()

        self.input_shape = input_shape
        self.embed_dim = embed_dim
        ntap, channels_2m = input_shape

        # CNN Branch
        # Process as 2D feature map: (Ntap, 2M)
        cnn_layers = []

        # First conv block
        cnn_layers.extend([
            nn.Conv2d(1, cnn_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_channels[0]),
            nn.ReLU(inplace=True)
        ])

        # Additional conv blocks with downsampling
        for i in range(len(cnn_channels) - 1):
            cnn_layers.extend([
                nn.Conv2d(cnn_channels[i], cnn_channels[i+1], kernel_size=3,
                         stride=2, padding=1),
                nn.BatchNorm2d(cnn_channels[i+1]),
                nn.ReLU(inplace=True)
            ])

        # Global average pooling
        cnn_layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.cnn_branch = nn.Sequential(*cnn_layers)
        cnn_output_dim = cnn_channels[-1]

        # MLP Branch
        mlp_input_dim = ntap * channels_2m
        self.mlp_branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(mlp_input_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Fusion layer
        fusion_input_dim = cnn_output_dim + mlp_hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch, Ntap, 2M) tensor

        Returns:
            (batch, embed_dim) tensor
        """
        batch_size = x.shape[0]

        # CNN branch: Add channel dimension for Conv2D
        x_cnn = x.unsqueeze(1)  # (batch, 1, Ntap, 2M)
        cnn_feat = self.cnn_branch(x_cnn)  # (batch, cnn_channels[-1], 1, 1)
        cnn_feat = cnn_feat.view(batch_size, -1)  # (batch, cnn_channels[-1])

        # MLP branch
        mlp_feat = self.mlp_branch(x)  # (batch, mlp_hidden_dim)

        # Concatenate features
        combined_feat = torch.cat([cnn_feat, mlp_feat], dim=1)  # (batch, fusion_input_dim)

        # Fusion to embedding
        embedding = self.fusion(combined_feat)  # (batch, embed_dim)

        return embedding


if __name__ == '__main__':
    # Test the encoder
    encoder = HybridEncoder(input_shape=(48, 32), embed_dim=256)
    x = torch.randn(8, 48, 32)  # batch_size=8
    out = encoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Expected output shape: (8, 256)")
    assert out.shape == (8, 256), "Output shape mismatch!"
    print("Encoder test passed!")
