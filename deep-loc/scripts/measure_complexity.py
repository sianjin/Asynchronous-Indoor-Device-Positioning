"""
Measure parameters, multiply-accumulate operations (MACs) and CPU inference
latency of the compared models, as a function of the number of anchors.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.task_configs import get_config
from src.models.localization_model import IndoorLocalizationModel
from src.models.baseline_cnn import EarlyFusionCNN


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Measure model complexity')
    parser.add_argument('--num_anchors', type=int, nargs='+', default=[4, 8, 16],
                       help='Numbers of anchors to evaluate')
    parser.add_argument('--repeats', type=int, default=200, help='Number of timed forward passes')
    return parser.parse_args()


def count_macs(model, x):
    """
    Count multiply-accumulate operations of one forward pass.

    Convolution and linear layers are counted with forward hooks. A Transformer
    encoder layer is counted analytically, because its inference fast path does
    not call its submodules: per token, 4 d^2 for the query, key, value and
    output projections, 2 Na d for the attention weights and 2 d d_ff for the
    feed-forward network.
    """
    macs = [0]

    def conv_hook(module, inputs, output):
        kernel = module.kernel_size[0] * module.kernel_size[1] * module.in_channels // module.groups
        macs[0] += output.numel() * kernel

    def linear_hook(module, inputs, output):
        macs[0] += output.numel() * module.in_features

    def transformer_layer_hook(module, inputs, output):
        batch, num_tokens, dim = inputs[0].shape
        ff_dim = module.linear1.out_features
        macs[0] += batch * num_tokens * (4 * dim * dim + 2 * num_tokens * dim + 2 * dim * ff_dim)

    # Submodules of a Transformer encoder layer are covered by the layer hook
    inside_transformer = {m for layer in model.modules() if isinstance(layer, nn.TransformerEncoderLayer)
                          for m in layer.modules() if m is not layer}

    hooks = []
    for module in model.modules():
        if module in inside_transformer:
            continue
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.TransformerEncoderLayer):
            hooks.append(module.register_forward_hook(transformer_layer_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    with torch.no_grad():
        model(x)
    for hook in hooks:
        hook.remove()
    return macs[0]


def measure_latency(model, x, repeats):
    """Median single-sample, single-thread CPU latency in milliseconds."""
    with torch.no_grad():
        for _ in range(10):
            model(x)
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - start)
    return 1e3 * float(np.median(times))


def main():
    """Main measurement function."""
    args = parse_args()
    torch.set_num_threads(1)

    print(f"{'Model':28s} {'Na':>3s} {'Params':>10s} {'MACs (M)':>10s} {'Latency (ms)':>13s}")
    print("-" * 70)
    for num_anchors in args.num_anchors:
        config = get_config('regression')
        config.num_anchors = num_anchors
        config.use_anchor_position = True
        config.anchor_positions = [[0.0, 0.0, 0.0]] * num_anchors

        deepsets_config = get_config('regression', num_layers=0)
        deepsets_config.use_anchor_position = True
        deepsets_config.anchor_positions = config.anchor_positions

        models = {
            'Proposed (CNN + Transformer)': IndoorLocalizationModel(config),
            'DeepSets (CNN + mean pool)': IndoorLocalizationModel(deepsets_config),
            'Early-fusion CNN': EarlyFusionCNN(input_shape=config.input_shape, num_anchors=num_anchors)
        }

        x = torch.randn(1, num_anchors, *config.input_shape)
        for name, model in models.items():
            model.eval()
            num_params = sum(p.numel() for p in model.parameters())
            if isinstance(model, IndoorLocalizationModel):
                # The unused classification head is not part of the regression model
                num_params -= sum(p.numel() for p in model.classification_head.parameters())
            print(f"{name:28s} {num_anchors:3d} {num_params:10,d} "
                  f"{count_macs(model, x) / 1e6:10.1f} {measure_latency(model, x, args.repeats):13.2f}")


if __name__ == '__main__':
    main()
