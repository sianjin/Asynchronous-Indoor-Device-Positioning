"""
Complete end-to-end indoor localization model.
Integrates encoder, transformer, and task-specific heads.
"""

import torch
import torch.nn as nn

from src.models.encoder import HybridEncoder
from src.models.transformer import CrossAnchorTransformer
from src.models.heads import RegressionHead, ClassificationHead


class IndoorLocalizationModel(nn.Module):
    """
    Complete indoor localization model.

    Architecture:
    1. Shared per-anchor encoder (Hybrid CNN + MLP)
    2. Cross-anchor transformer fusion
    3. Global aggregation (mean pooling)
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

        # Shared per-anchor encoder
        self.encoder = HybridEncoder(
            input_shape=config.input_shape,
            embed_dim=config.embed_dim,
            cnn_channels=config.cnn_channels,
            mlp_hidden_dim=config.mlp_hidden_dim,
            dropout=config.encoder_dropout
        )

        # Cross-anchor transformer
        self.transformer = CrossAnchorTransformer(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            ff_dim=config.ff_dim,
            dropout=config.transformer_dropout
        )

        # Task-specific heads
        self.regression_head = RegressionHead(
            input_dim=config.embed_dim,
            hidden_dims=config.regression_hidden_dims,
            dropout=config.head_dropout
        )

        self.classification_head = ClassificationHead(
            input_dim=config.embed_dim,
            hidden_dims=config.classification_hidden_dims,
            num_classes=config.num_classes,
            dropout=config.head_dropout
        )

    def forward(self, anchor_features):
        """
        Forward pass.

        Args:
            anchor_features: (batch, Na, Ntap, 2M) tensor

        Returns:
            (batch, 3) for regression or (batch, num_classes) for classification
        """
        batch_size, Na, Ntap, channels_2M = anchor_features.shape

        # Step 1: Encode each anchor independently using shared encoder
        # Reshape to (batch*Na, Ntap, 2M) to process all anchors in parallel
        flat_features = anchor_features.reshape(batch_size * Na, Ntap, channels_2M)

        # Apply encoder
        embeddings = self.encoder(flat_features)  # (batch*Na, embed_dim)

        # Step 2: Reshape back to (batch, Na, embed_dim)
        anchor_embeddings = embeddings.reshape(batch_size, Na, -1)

        # Step 3: Cross-anchor fusion via transformer
        fused_embeddings = self.transformer(anchor_embeddings)  # (batch, Na, embed_dim)

        # Step 4: Global aggregation via mean pooling (permutation-invariant)
        global_feature = fused_embeddings.mean(dim=1)  # (batch, embed_dim)

        # Step 5: Task-specific head
        if self.task == 'regression':
            output = self.regression_head(global_feature)  # (batch, 3)
        elif self.task == 'classification':
            output = self.classification_head(global_feature)  # (batch, num_classes)
        else:
            raise ValueError(f"Unknown task: {self.task}")

        return output

    def predict(self, anchor_features):
        """
        Inference mode with no gradient computation.

        Args:
            anchor_features: (batch, Na, Ntap, 2M) tensor

        Returns:
            predictions
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(anchor_features)

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

    print("\nTesting Classification Model:")
    cls_config = ClassificationConfig()
    cls_model = IndoorLocalizationModel(cls_config)

    out = cls_model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Expected: (8, 100)")
    assert out.shape == (8, 100), "Classification model output shape mismatch!"
    print("  Test passed!")

    # Test prediction mode
    pred = cls_model.predict(x)
    assert pred.shape == (8,), "Prediction shape mismatch!"
    print("  Prediction mode test passed!")

    # Count parameters
    total_params = sum(p.numel() for p in reg_model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
