"""
Early-fusion CNN baseline.
PyTorch port of the network in the MathWorks 802.11az fingerprinting
example: the anchors are stacked as input channels.
"""

import torch
import torch.nn as nn


class EarlyFusionCNN(nn.Module):
    """
    CNN baseline that treats the anchors as image channels.

    Unlike IndoorLocalizationModel, the output depends on the anchor order and
    a missing anchor can only be represented by an all-zero channel.

    Input: (batch, Na, Ntap, 2M)
    Output: (batch, 3)
    """

    def __init__(self, input_shape=(48, 32), num_anchors=4, num_filters=256,
                 num_blocks=4, dropout=0.2):
        """
        Initialize baseline CNN.

        Args:
            input_shape: (Ntap, 2M) - input feature map shape
            num_anchors: number of anchors (input channels)
            num_filters: number of filters in every convolution layer
            num_blocks: number of convolution blocks
            dropout: dropout probability before the output layer
        """
        super().__init__()

        layers = []
        in_channels = num_anchors
        height, width = input_shape
        for _ in range(num_blocks):
            layers.extend([
                nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(2, stride=2, ceil_mode=True)
            ])
            in_channels = num_filters
            height, width = -(-height // 2), -(-width // 2)

        self.features = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * height * width, 3)

    def forward(self, anchor_features, anchor_mask=None):
        """
        Forward pass.

        Args:
            anchor_features: (batch, Na, Ntap, 2M) tensor
            anchor_mask: optional (batch, Na) bool tensor, True for anchors to ignore

        Returns:
            (batch, 3) tensor of (x, y, z) positions
        """
        if anchor_mask is not None:
            anchor_features = anchor_features.masked_fill(anchor_mask[:, :, None, None], 0.0)

        x = self.features(anchor_features)
        x = self.dropout(x.flatten(1))
        return self.fc(x)


if __name__ == '__main__':
    model = EarlyFusionCNN()
    x = torch.randn(8, 4, 48, 32)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (8, 3), "Output shape mismatch!"
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
