"""
Inference script for trained models.
"""

import os
import sys
import argparse
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.task_configs import get_config
from src.models.localization_model import IndoorLocalizationModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference with trained model')

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input data (.npy file with shape (Na, Ntap, 2M))')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')

    return parser.parse_args()


def load_model(checkpoint_path, device):
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: path to checkpoint
        device: device to load model on

    Returns:
        model, config
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint
    config_dict = checkpoint.get('config', {})
    task = config_dict.get('task', 'regression')

    # Create config
    config = get_config(task)

    # Update config with checkpoint values
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Create model
    model = IndoorLocalizationModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded successfully")
    print(f"Task: {config.task}")
    print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")

    return model, config


def run_inference(model, input_data, config, device):
    """
    Run inference on input data.

    Args:
        model: trained model
        input_data: input features (Na, Ntap, 2M) or (batch, Na, Ntap, 2M)
        config: configuration object
        device: device to use

    Returns:
        predictions
    """
    # Add batch dimension if needed
    if input_data.ndim == 3:
        input_data = input_data[np.newaxis, ...]  # (1, Na, Ntap, 2M)

    # Convert to tensor
    input_tensor = torch.from_numpy(input_data).float().to(device)

    # Run inference
    with torch.no_grad():
        predictions = model.predict(input_tensor)

    return predictions.cpu().numpy()


def main():
    """Main inference function."""
    args = parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model, config = load_model(args.checkpoint, device)

    # Load input data
    print(f"\nLoading input data: {args.input}")
    input_data = np.load(args.input)
    print(f"Input shape: {input_data.shape}")

    # Validate input shape
    expected_shape = (config.num_anchors, *config.input_shape)
    if input_data.shape[-3:] != expected_shape:
        raise ValueError(
            f"Input shape mismatch. Expected (..., {expected_shape}), "
            f"got {input_data.shape}"
        )

    # Run inference
    print("\nRunning inference...")
    predictions = run_inference(model, input_data, config, device)

    # Print results
    print("\n" + "=" * 80)
    print("INFERENCE RESULTS")
    print("=" * 80)

    if config.task == 'regression':
        print(f"Number of samples: {predictions.shape[0]}")
        print(f"\nPredicted 3D positions (x, y, z):")
        print("-" * 80)

        # If predictions were normalized, denormalize them
        if config.normalize_positions:
            pos_min = np.array(config.pos_min)
            pos_max = np.array(config.pos_max)
            predictions_denorm = predictions * (pos_max - pos_min) + pos_min

            for i, pred in enumerate(predictions_denorm):
                print(f"Sample {i}: x={pred[0]:.3f}m, y={pred[1]:.3f}m, z={pred[2]:.3f}m")
        else:
            for i, pred in enumerate(predictions):
                print(f"Sample {i}: x={pred[0]:.3f}m, y={pred[1]:.3f}m, z={pred[2]:.3f}m")

    elif config.task == 'classification':
        print(f"Number of samples: {len(predictions)}")
        print(f"\nPredicted classes:")
        print("-" * 80)

        for i, pred in enumerate(predictions):
            print(f"Sample {i}: Class {pred}")

    print("=" * 80)


if __name__ == '__main__':
    main()
