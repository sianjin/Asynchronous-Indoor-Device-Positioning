"""
Cross-anchor transformer for fusing information across anchors.
"""

import torch
import torch.nn as nn


class CrossAnchorTransformer(nn.Module):
    """
    Transformer encoder for cross-anchor fusion via self-attention.

    Uses standard Transformer encoder architecture without positional encoding
    to maintain permutation invariance across anchors.

    Input: (batch, Na, embed_dim)
    Output: (batch, Na, embed_dim)
    """

    def __init__(self, embed_dim=256, num_heads=4, num_layers=2, ff_dim=512, dropout=0.1):
        """
        Initialize cross-anchor transformer.

        Args:
            embed_dim: embedding dimension
            num_heads: number of attention heads
            num_layers: number of transformer layers
            ff_dim: feed-forward network hidden dimension
            dropout: dropout probability
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True,  # Important: batch dimension comes first
            norm_first=False
        )

        # Stack multiple layers
        # Nested-tensor fast path is disabled: with a key padding mask it
        # truncates the output to the longest unmasked sequence in the batch
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False
        ) if num_layers > 0 else None

        # No positional encoding - we want permutation invariance

    def forward(self, x, key_padding_mask=None):
        """
        Forward pass.

        Args:
            x: (batch, Na, embed_dim) tensor of anchor embeddings
            key_padding_mask: optional (batch, Na) bool tensor, True for
                missing anchors that must not be attended to

        Returns:
            (batch, Na, embed_dim) tensor of fused embeddings
        """
        # num_layers = 0 disables cross-anchor fusion (DeepSets-style baseline)
        if self.transformer_encoder is None:
            return x

        # Apply transformer encoder
        # Note: No positional encoding is added to maintain permutation invariance
        out = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)

        return out


if __name__ == '__main__':
    # Test the transformer
    transformer = CrossAnchorTransformer(embed_dim=256, num_heads=4, num_layers=2)
    x = torch.randn(8, 4, 256)  # batch_size=8, num_anchors=4, embed_dim=256
    out = transformer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Expected output shape: (8, 4, 256)")
    assert out.shape == (8, 4, 256), "Output shape mismatch!"
    print("Transformer test passed!")

    # Test permutation invariance (output order should change with input order)
    x_perm = x[:, [1, 0, 3, 2], :]  # Permute anchors
    out_perm = transformer(x_perm)
    # The outputs should be permuted in the same way
    print("Permutation test: outputs should differ for different anchor orders")
