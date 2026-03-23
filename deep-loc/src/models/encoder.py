"""
CNN-based token encoder for multi-token per-anchor architecture.
Matches paper Section III.1 design.
"""

import torch
import torch.nn as nn


class CNNTokenEncoder(nn.Module):
    """
    CNN encoder that outputs multiple spatial tokens per anchor.

    Architecture:
    - 3-layer CNN with stride-2 downsampling
    - No global pooling - preserves spatial structure
    - Token projection to embedding dimension

    Input: (batch, Ntap, 2M) = (batch, 48, 32)
    Output: (batch, P, d) where P = Nf × Mf, d = embed_dim
    """

    def __init__(self, input_shape=(48, 32), embed_dim=256,
                 cnn_channels=[32, 64, 128]):
        """
        Initialize CNN token encoder.

        Args:
            input_shape: (Ntap, 2M) input dimensions
            embed_dim: output token embedding dimension
            cnn_channels: channel progression for CNN layers
        """
        super().__init__()

        self.input_shape = input_shape
        self.embed_dim = embed_dim

        # Build CNN layers
        cnn_layers = []

        # First conv: no downsampling
        cnn_layers.extend([
            nn.Conv2d(1, cnn_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_channels[0]),
            nn.ReLU(inplace=True)
        ])

        # Subsequent convs: stride-2 downsampling
        for i in range(len(cnn_channels) - 1):
            cnn_layers.extend([
                nn.Conv2d(cnn_channels[i], cnn_channels[i+1],
                         kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(cnn_channels[i+1]),
                nn.ReLU(inplace=True)
            ])

        self.cnn = nn.Sequential(*cnn_layers)

        # Token projection: Cf → d
        self.token_proj = nn.Linear(cnn_channels[-1], embed_dim)

        # Calculate output spatial dimensions
        # After 2 stride-2 convolutions: (48, 32) → (24, 16) → (12, 8)
        self.tokens_per_anchor = 12 * 8  # 96 tokens

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch, Ntap, 2M) tensor

        Returns:
            (batch, P, d) tensor of token embeddings
        """
        batch_size = x.shape[0]

        # Add channel dimension for Conv2d
        x = x.unsqueeze(1)  # (batch, 1, Ntap, 2M)

        # Extract spatial features via CNN
        features = self.cnn(x)  # (batch, Cf, Nf, Mf)

        # Reshape to tokens: (batch, P, Cf) where P = Nf × Mf
        # features.flatten(2): (batch, Cf, Nf*Mf)
        # .transpose(1, 2): (batch, Nf*Mf, Cf)
        tokens = features.flatten(2).transpose(1, 2)  # (batch, P, Cf)

        # Project to embedding dimension
        token_embeddings = self.token_proj(tokens)  # (batch, P, d)

        return token_embeddings


if __name__ == '__main__':
    # Test the encoder
    encoder = CNNTokenEncoder(input_shape=(48, 32), embed_dim=256)
    x = torch.randn(8, 48, 32)  # batch_size=8
    out = encoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Expected: (8, 96, 256)")
    print(f"Tokens per anchor: {encoder.tokens_per_anchor}")
    assert out.shape == (8, 96, 256), "Output shape mismatch!"
    print("✓ CNNTokenEncoder test passed!")
