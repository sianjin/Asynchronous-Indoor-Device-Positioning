"""
CNN encoder for per-anchor feature extraction.
"""

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """
    Pure CNN encoder with global average pooling.

    Architecture:
    - CNN feature extraction via 2D convolutions over (Ntap, 2M) delay-spatial map
    - Global average pooling to aggregate spatial information
    - Linear projection to embedding dimension

    Input: (batch, Ntap, 2M) = (batch, 48, 32)
    Output: (batch, embed_dim)
    """

    def __init__(self, input_shape=(48, 32), embed_dim=256,
                 cnn_channels=[64, 128, 256], dropout=0.15):
        """
        Initialize CNN encoder.

        Args:
            input_shape: (Ntap, 2M) - input feature map shape
            embed_dim: output embedding dimension
            cnn_channels: list of channel dimensions for CNN layers
            dropout: dropout probability
        """
        super().__init__()

        self.input_shape = input_shape
        self.embed_dim = embed_dim

        # CNN feature extraction
        cnn_layers = []

        # First conv block (no downsampling to preserve spatial info)
        cnn_layers.extend([
            nn.Conv2d(1, cnn_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_channels[0]),
            nn.ReLU(inplace=True)
        ])

        # Additional conv blocks with stride-2 downsampling
        for i in range(len(cnn_channels) - 1):
            cnn_layers.extend([
                nn.Conv2d(cnn_channels[i], cnn_channels[i+1], kernel_size=3,
                         stride=2, padding=1),
                nn.BatchNorm2d(cnn_channels[i+1]),
                nn.ReLU(inplace=True)
            ])

        # Global average pooling
        # Aggregates spatial dimensions: (batch, Cf, Nf, Mf) → (batch, Cf, 1, 1)
        cnn_layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.cnn = nn.Sequential(*cnn_layers)

        # Linear projection to embedding dimension
        # Projects pooled features: (batch, Cf) → (batch, embed_dim)
        self.projection = nn.Sequential(
            nn.Linear(cnn_channels[-1], embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch, Ntap, 2M) tensor

        Returns:
            (batch, embed_dim) tensor - single embedding per anchor
        """
        batch_size = x.shape[0]

        # Add channel dimension for Conv2d
        x = x.unsqueeze(1)  # (batch, 1, Ntap, 2M)

        # Extract features via CNN
        features = self.cnn(x)  # (batch, Cf, 1, 1)

        # Flatten spatial dimensions after global pooling
        h = features.view(batch_size, -1)  # (batch, Cf)

        # Project to embedding dimension
        embedding = self.projection(h)  # (batch, embed_dim)

        return embedding


if __name__ == '__main__':
    # Test the CNN encoder
    print("Testing CNNEncoder...")
    encoder = CNNEncoder(input_shape=(48, 32), embed_dim=256)
    x = torch.randn(8, 48, 32)  # batch_size=8
    out = encoder(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Expected output shape: (8, 256)")

    # Count parameters
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"Number of parameters: {num_params:,}")

    assert out.shape == (8, 256), "Output shape mismatch!"
    print("✓ CNNEncoder test passed!")
