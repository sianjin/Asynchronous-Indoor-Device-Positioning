"""
Complete end-to-end indoor localization model.
Integrates encoder, transformer, and task-specific heads.
"""

import torch
import torch.nn as nn

from src.models.encoder import CNNEncoder
from src.models.transformer import CrossAnchorTransformer
from src.models.heads import RegressionHead, ClassificationHead


class IndoorLocalizationModel(nn.Module):
    """
    Complete indoor localization model.

    Architecture:
    1. Shared per-anchor encoder (Pure CNN with global pooling)
    2. Cross-anchor transformer fusion (missing anchors are masked)
    3. Global aggregation (masked mean pooling, attention pooling, or
       concatenation in the fixed anchor order as an order-dependent baseline)
    4. Task-specific head (regression or classification)

    Input: (batch, Na, Ntap, 2M)
    Output: (batch, 3) for regression or (batch, num_classes) for classification
    """

    def __init__(self, config):
        """
        Initialize localization model.

        Args:
            config: configuration object with all hyperparameters
        """
        super().__init__()

        self.task = config.task
        self.num_anchors = config.num_anchors
        self.mask_missing_anchors = getattr(config, 'mask_missing_anchors', True)
        self.anchor_dropout = getattr(config, 'anchor_dropout', 0.0)
        self.pooling = getattr(config, 'pooling', 'mean')
        self.use_anchor_position = getattr(config, 'use_anchor_position', False)

        # Shared per-anchor encoder
        self.encoder = CNNEncoder(
            input_shape=config.input_shape,
            embed_dim=config.embed_dim,
            cnn_channels=config.cnn_channels,
            dropout=config.encoder_dropout
        )

        # Optional embedding of the AP coordinates, added to each anchor token.
        # The input stays a set of (CIR, AP position) pairs, so the model
        # remains permutation invariant
        if self.use_anchor_position:
            pos_min = torch.tensor(config.pos_min, dtype=torch.float32)
            pos_max = torch.tensor(config.pos_max, dtype=torch.float32)
            anchor_pos = torch.tensor(config.anchor_positions, dtype=torch.float32)
            self.register_buffer('anchor_pos', (anchor_pos - pos_min) / (pos_max - pos_min))
            self.anchor_pos_embed = nn.Sequential(
                nn.Linear(3, config.embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(config.embed_dim, config.embed_dim)
            )

        # Cross-anchor transformer
        self.transformer = CrossAnchorTransformer(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            ff_dim=config.ff_dim,
            dropout=config.transformer_dropout
        )

        # Attention pooling: learned scalar score per anchor
        if self.pooling == 'attention':
            self.pool_score = nn.Linear(config.embed_dim, 1)
        elif self.pooling not in ('mean', 'concat'):
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # Concatenation keeps one slot per anchor, so the head grows with Na
        head_input_dim = config.embed_dim * (config.num_anchors if self.pooling == 'concat' else 1)

        # Task-specific heads
        self.regression_head = RegressionHead(
            input_dim=head_input_dim,
            hidden_dims=config.regression_hidden_dims,
            dropout=config.head_dropout
        )

        self.classification_head = ClassificationHead(
            input_dim=head_input_dim,
            hidden_dims=config.classification_hidden_dims,
            num_classes=config.num_classes,
            dropout=config.head_dropout
        )

    def _anchor_mask(self, anchor_features, anchor_mask):
        """
        Build the (batch, Na) mask of missing anchors (True = missing).

        An anchor is missing if its observation is all-zero (no packet was
        detected), if the caller marks it in anchor_mask, or if it is dropped
        by anchor dropout during training. At least one anchor is always kept.
        """
        batch_size, Na = anchor_features.shape[:2]
        if self.mask_missing_anchors:
            missing = anchor_features.reshape(batch_size, Na, -1).abs().sum(dim=-1) == 0
        else:
            missing = torch.zeros(batch_size, Na, dtype=torch.bool, device=anchor_features.device)

        if anchor_mask is not None:
            missing = missing | anchor_mask

        if self.training and self.anchor_dropout > 0:
            drop = torch.rand(batch_size, Na, device=anchor_features.device) < self.anchor_dropout
            dropped = missing | drop
            # Do not apply dropout to samples that would lose every anchor
            keep_original = dropped.all(dim=1, keepdim=True)
            missing = torch.where(keep_original, missing, dropped)

        # A sample with no anchors at all cannot be masked (attention would be NaN)
        missing = missing & ~missing.all(dim=1, keepdim=True)
        return missing

    def forward(self, anchor_features, anchor_mask=None, anchor_positions=None):
        """
        Forward pass.

        Args:
            anchor_features: (batch, Na, Ntap, 2M) tensor
            anchor_mask: optional (batch, Na) bool tensor, True for anchors to ignore
            anchor_positions: optional (batch, Na, 3) tensor of AP coordinates
                normalized to the room dimensions, for data in which the AP
                layout changes from sample to sample. By default the fixed
                layout of the configuration is used

        Returns:
            (batch, 3) for regression or (batch, num_classes) for classification
        """
        batch_size, Na, Ntap, channels_2M = anchor_features.shape

        missing = self._anchor_mask(anchor_features, anchor_mask)

        # Step 1: Encode each anchor independently using shared encoder
        # Reshape to (batch*Na, Ntap, 2M) to process all anchors in parallel
        flat_features = anchor_features.reshape(batch_size * Na, Ntap, channels_2M)

        # Apply encoder
        embeddings = self.encoder(flat_features)  # (batch*Na, embed_dim)

        # Step 2: Reshape back to (batch, Na, embed_dim)
        anchor_embeddings = embeddings.reshape(batch_size, Na, -1)

        if self.use_anchor_position:
            if anchor_positions is None:
                anchor_positions = self.anchor_pos[:Na]
            anchor_embeddings = anchor_embeddings + self.anchor_pos_embed(anchor_positions)

        # Step 3: Cross-anchor fusion via transformer
        fused_embeddings = self.transformer(anchor_embeddings, key_padding_mask=missing)  # (batch, Na, embed_dim)

        # Step 4: Global aggregation over the available anchors (permutation-invariant)
        if self.pooling == 'concat':
            # Order-dependent baseline: missing anchors leave an all-zero slot
            global_feature = fused_embeddings.masked_fill(missing.unsqueeze(-1), 0.0).flatten(1)
        else:
            if self.pooling == 'attention':
                scores = self.pool_score(fused_embeddings).squeeze(-1)  # (batch, Na)
                weights = scores.masked_fill(missing, float('-inf')).softmax(dim=1)
            else:
                weights = (~missing).float()
                weights = weights / weights.sum(dim=1, keepdim=True)
            global_feature = (fused_embeddings * weights.unsqueeze(-1)).sum(dim=1)  # (batch, embed_dim)

        # Step 5: Task-specific head
        if self.task == 'regression':
            output = self.regression_head(global_feature)  # (batch, 3)
        elif self.task == 'classification':
            output = self.classification_head(global_feature)  # (batch, num_classes)
        else:
            raise ValueError(f"Unknown task: {self.task}")

        return output

    def predict(self, anchor_features, anchor_mask=None, anchor_positions=None):
        """
        Inference mode with no gradient computation.

        Args:
            anchor_features: (batch, Na, Ntap, 2M) tensor
            anchor_mask: optional (batch, Na) bool tensor, True for anchors to ignore

        Returns:
            predictions
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(anchor_features, anchor_mask, anchor_positions)

        if self.task == 'classification':
            # Return class predictions
            return output.argmax(dim=-1)
        else:
            # Return regression predictions
            return output


if __name__ == '__main__':
    # Test the complete model
    from config.task_configs import RegressionConfig, ClassificationConfig

    print("Testing Regression Model:")
    reg_config = RegressionConfig()
    reg_model = IndoorLocalizationModel(reg_config)

    # Input: (batch, Na, Ntap, 2M) = (8, 4, 48, 32)
    x = torch.randn(8, 4, 48, 32)
    out = reg_model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Expected: (8, 3)")
    assert out.shape == (8, 3), "Regression model output shape mismatch!"
    print("  Test passed!")

    # Test prediction mode
    pred = reg_model.predict(x)
    assert pred.shape == (8, 3), "Prediction shape mismatch!"
    print("  Prediction mode test passed!")

    # Test permutation invariance and masking of missing anchors
    perm = [2, 0, 3, 1]
    assert torch.allclose(pred, reg_model.predict(x[:, perm]), atol=1e-5), "Not permutation invariant!"
    x_missing = x.clone()
    x_missing[:, 1] = 0
    assert torch.allclose(reg_model.predict(x_missing), reg_model.predict(x[:, [0, 2, 3]]), atol=1e-5), \
        "Missing anchor is not masked!"
    print("  Permutation invariance and masking tests passed!")

    print("\nTesting Classification Model:")
    cls_config = ClassificationConfig()
    cls_model = IndoorLocalizationModel(cls_config)

    out = cls_model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Expected: (8, {cls_config.num_classes})")
    assert out.shape == (8, cls_config.num_classes), "Classification model output shape mismatch!"
    print("  Test passed!")

    # Test prediction mode
    pred = cls_model.predict(x)
    assert pred.shape == (8,), "Prediction shape mismatch!"
    print("  Prediction mode test passed!")

    # Count parameters
    total_params = sum(p.numel() for p in reg_model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
