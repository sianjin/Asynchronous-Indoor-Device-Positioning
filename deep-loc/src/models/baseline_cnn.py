"""
Early-fusion CNN baselines: the anchors are stacked as input channels.

EarlyFusionCNN is a PyTorch port of the network in the MathWorks 802.11az
fingerprinting example. EarlyFusionResNet is a deeper residual network of the
kind used for CSI fingerprinting.
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
                 num_blocks=4, dropout=0.2, anchor_dropout=0.0, use_anchor_position=False):
        """
        Initialize baseline CNN.

        Args:
            input_shape: (Ntap, 2M) - input feature map shape
            num_anchors: number of anchors (input channels)
            num_filters: number of filters in every convolution layer
            num_blocks: number of convolution blocks
            dropout: dropout probability before the output layer
            anchor_dropout: probability of zeroing each anchor during training
            use_anchor_position: append an embedding of the AP coordinates to the features
        """
        super().__init__()

        self.anchor_dropout = anchor_dropout
        self.position_branch = AnchorPositionBranch(num_anchors) if use_anchor_position else None

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
        self.fc = nn.Linear(num_filters * height * width
                            + (self.position_branch.dim if use_anchor_position else 0), 3)

    def forward(self, anchor_features, anchor_mask=None, anchor_positions=None):
        """
        Forward pass.

        Args:
            anchor_features: (batch, Na, Ntap, 2M) tensor
            anchor_mask: optional (batch, Na) bool tensor, True for anchors to ignore

        Returns:
            (batch, 3) tensor of (x, y, z) positions
        """
        anchor_features = remove_anchors(anchor_features, anchor_mask,
                                         self.anchor_dropout if self.training else 0.0)

        x = self.features(anchor_features)
        x = self.dropout(x.flatten(1))
        if self.position_branch is not None:
            x = torch.cat([x, self.position_branch(anchor_features, anchor_positions)], dim=1)
        return self.fc(x)


class AnchorPositionBranch(nn.Module):
    """
    Embedding of the AP coordinates for an early-fusion network.

    The coordinates of all anchors are concatenated in the anchor order
    together with a flag that marks the anchors that are present, embedded by
    an MLP and appended to the image features before the output layer.
    """

    def __init__(self, num_anchors, dim=64):
        super().__init__()

        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(4 * num_anchors, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, anchor_features, anchor_positions):
        """
        Args:
            anchor_features: (batch, Na, Ntap, 2M) tensor; all-zero anchors are absent
            anchor_positions: (batch, Na, 3) tensor of normalized AP coordinates

        Returns:
            (batch, dim) tensor
        """
        present = (anchor_features.flatten(2).abs().sum(dim=-1) > 0).float().unsqueeze(-1)
        return self.mlp(torch.cat([anchor_positions * present, present], dim=-1).flatten(1))


def remove_anchors(anchor_features, anchor_mask=None, anchor_dropout=0.0):
    """
    Zero the channels of the anchors in anchor_mask and of randomly dropped anchors.

    An early-fusion network has no other way to represent a missing anchor.
    """
    if anchor_mask is not None:
        anchor_features = anchor_features.masked_fill(anchor_mask[:, :, None, None], 0.0)
    if anchor_dropout > 0:
        drop = torch.rand(anchor_features.shape[:2], device=anchor_features.device) < anchor_dropout
        # Do not apply dropout to samples that would lose every anchor
        drop = drop & ~drop.all(dim=1, keepdim=True)
        anchor_features = anchor_features.masked_fill(drop[:, :, None, None], 0.0)
    return anchor_features


class BasicBlock(nn.Module):
    """Residual block with two 3x3 convolutions."""

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return torch.relu(self.conv(x) + self.shortcut(x))


class EarlyFusionResNet(nn.Module):
    """
    Residual network baseline with the ResNet-18 layout (two blocks per stage).

    Input: (batch, Na, Ntap, 2M)
    Output: (batch, 3)
    """

    def __init__(self, num_anchors=4, widths=[32, 64, 128, 256], dropout=0.2,
                 use_anchor_position=False):
        """
        Initialize residual baseline.

        Args:
            num_anchors: number of anchors (input channels)
            widths: number of channels of the four stages
            dropout: dropout probability before the output layer
            use_anchor_position: append an embedding of the AP coordinates to the features
        """
        super().__init__()

        self.position_branch = AnchorPositionBranch(num_anchors) if use_anchor_position else None

        layers = [
            nn.Conv2d(num_anchors, widths[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True)
        ]
        in_channels = widths[0]
        for i, width in enumerate(widths):
            layers.append(BasicBlock(in_channels, width, stride=1 if i == 0 else 2))
            layers.append(BasicBlock(width, width, stride=1))
            in_channels = width
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.features = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels + (self.position_branch.dim if use_anchor_position else 0), 3)

    def forward(self, anchor_features, anchor_mask=None, anchor_positions=None):
        """
        Forward pass.

        Args:
            anchor_features: (batch, Na, Ntap, 2M) tensor
            anchor_mask: optional (batch, Na) bool tensor, True for anchors to ignore

        Returns:
            (batch, 3) tensor of (x, y, z) positions
        """
        anchor_features = remove_anchors(anchor_features, anchor_mask)

        x = self.features(anchor_features)
        x = self.dropout(x.flatten(1))
        if self.position_branch is not None:
            x = torch.cat([x, self.position_branch(anchor_features, anchor_positions)], dim=1)
        return self.fc(x)


if __name__ == '__main__':
    model = EarlyFusionCNN()
    x = torch.randn(8, 4, 48, 32)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (8, 3), "Output shape mismatch!"
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    resnet = EarlyFusionResNet()
    assert resnet(x).shape == (8, 3), "Output shape mismatch!"
    print(f"ResNet number of parameters: {sum(p.numel() for p in resnet.parameters()):,}")
