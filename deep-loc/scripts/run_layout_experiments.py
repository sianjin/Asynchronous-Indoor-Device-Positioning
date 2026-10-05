"""
Experiment runner for the multi-layout study.

The datasets written by phy/wifiPosGenerateDataLayouts.m contain many AP
layouts with 3 to 6 APs at random positions. A model is trained on the
training layouts, the checkpoint with the lowest error on held-out training
layouts is kept, and it is evaluated on

  - unseen layouts: AP layouts that are not in the training set, and
  - seen layouts:   training layouts with new device positions.

All models share the same split, optimizer, schedule and random receiver
reference (random sub-sample delay and phase per AP).
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import scipy.io
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.task_configs import get_config
from src.models.localization_model import IndoorLocalizationModel
from src.models.baseline_cnn import EarlyFusionCNN, EarlyFusionResNet
from run_experiments import randomize_reference


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run one experiment of the multi-layout study')

    parser.add_argument('--name', type=str, required=True, help='Experiment name')
    parser.add_argument('--model', type=str, default='transformer',
                       choices=['transformer', 'cnn', 'resnet', 'knn'], help='Model type')
    parser.add_argument('--seed', type=int, default=0, help='Seed for the split and the training')
    parser.add_argument('--data_dir', type=str, default='.',
                       help='Directory with the data_layouts_*.mat files')
    parser.add_argument('--data_prefix', type=str, default='',
                       help='Prefix of the data files ("smoke_" for the MATLAB smoke test files)')
    parser.add_argument('--output_dir', type=str, default='results/layouts',
                       help='Directory to save results')

    # Model variants
    parser.add_argument('--num_layers', type=int, default=None,
                       help='Number of transformer layers (0 = no cross-anchor attention)')
    parser.add_argument('--pooling', type=str, default=None, choices=['mean', 'attention', 'concat'])
    parser.add_argument('--use_anchor_position', action='store_true',
                       help='Give the model the AP coordinates of every sample')

    # Training
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--warmup_epochs', type=int, default=None)
    parser.add_argument('--patience', type=int, default=None, help='Early stopping patience in epochs')
    parser.add_argument('--threads', type=int, default=None, help='Number of CPU threads')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'mps', 'cpu'],
                       help='Device to use (default: cuda if available, otherwise cpu)')
    parser.add_argument('--smoke', action='store_true',
                       help='Quick test of the pipeline: one epoch on a small subset of the data')

    return parser.parse_args()


def load_layout_file(path):
    """
    Load one data_layouts_*.mat file.

    Returns:
        dict of numpy arrays with one entry per sample:
            X (N, Na, Ntap, 2M), Y (N, 3), P (N, Na, 3) AP coordinates in meters,
            layout (N,), num_aps (N,), snr (N,)
        Absent and undetected APs are all-zero in X. The APs of every sample are
        sorted by their x and then y coordinate, with absent APs last, so that
        models that depend on the AP order see a consistent order.
    """
    data = scipy.io.loadmat(path)
    X = np.transpose(data['X'], (3, 2, 0, 1)).astype(np.float32)  # (N, Na, Ntap, 2M)
    P = np.transpose(data['apPositions'], (2, 1, 0)).astype(np.float32)  # (N, Na, 3)
    present = data['present'].T.astype(bool)  # (N, Na)

    key = np.where(present, np.round(P[..., 0], 3) * 1e3 + P[..., 1], np.inf)
    order = np.argsort(key, axis=1, kind='stable')
    X = np.take_along_axis(X, order[:, :, None, None], axis=1)
    P = np.take_along_axis(P, order[:, :, None], axis=1)

    return {
        'X': X, 'P': P,
        'Y': data['position'].T.astype(np.float32),
        'layout': data['layout'].reshape(-1).astype(np.int64),
        'num_aps': data['numAPs'].reshape(-1).astype(np.int64),
        'snr': data['snr'].reshape(-1)
    }


def load_layout_split(data_dir, prefix, seed, pos_min, pos_max, val_frac=0.15):
    """
    Load the datasets and split the training layouts into training and validation.

    Returns:
        dict: set name -> dict of tensors X, Y, P (normalized AP coordinates),
            num_aps, num_detected. Samples without a detected AP are removed.
    """
    train = load_layout_file(os.path.join(data_dir, f'{prefix}data_layouts_train.mat'))
    sets = {
        'test_unseen': load_layout_file(os.path.join(data_dir, f'{prefix}data_layouts_test_unseen.mat')),
        'test_seen': load_layout_file(os.path.join(data_dir, f'{prefix}data_layouts_test_seen.mat'))
    }

    # Validation layouts are held out from the training layouts, so the model
    # selection also targets layouts that are not trained on
    layouts = np.unique(train['layout'])
    order = np.random.RandomState(seed).permutation(len(layouts))
    val_layouts = layouts[order[:max(1, int(round(val_frac * len(layouts))))]]
    is_val = np.isin(train['layout'], val_layouts)
    sets['train'] = {k: v[~is_val] for k, v in train.items()}
    sets['val'] = {k: v[is_val] for k, v in train.items()}

    X_train = sets['train']['X']
    scale = X_train[np.abs(X_train).sum(axis=(2, 3)) > 0].std()

    split = {}
    for name, d in sets.items():
        detected = np.abs(d['X']).sum(axis=(2, 3)) > 0  # (N, Na)
        keep = detected.any(axis=1)
        split[name] = {
            'X': torch.from_numpy(d['X'][keep] / scale),
            'Y': torch.from_numpy(d['Y'][keep]),
            'P': (torch.from_numpy(d['P'][keep]) - pos_min) / (pos_max - pos_min),
            'num_aps': torch.from_numpy(d['num_aps'][keep]),
            'num_detected': torch.from_numpy(detected[keep].sum(axis=1)),
            'num_layouts': len(np.unique(d['layout']))
        }
    return split


def build_model(args, config):
    """Build the model for this experiment."""
    if args.model == 'cnn':
        return EarlyFusionCNN(input_shape=config.input_shape, num_anchors=config.num_anchors,
                              use_anchor_position=args.use_anchor_position)
    if args.model == 'resnet':
        return EarlyFusionResNet(num_anchors=config.num_anchors,
                                 use_anchor_position=args.use_anchor_position)
    return IndoorLocalizationModel(config)


def predict(model, d, batch_size=256):
    """Predict normalized positions for all samples of a set."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(d['X']), batch_size):
            preds.append(model(d['X'][i:i + batch_size], None, d['P'][i:i + batch_size]))
    return torch.cat(preds, dim=0)


def distance_error(pred_norm, target, pos_min, pos_max):
    """3D distance error in meters from normalized predictions."""
    pred = pred_norm * (pos_max - pos_min) + pos_min
    return torch.sqrt(((pred - target) ** 2).sum(dim=-1))


def train(model, split, config, pos_min, pos_max):
    """Train with warmup + cosine schedule; keep the best checkpoint on the validation layouts."""
    X_train, Y_train, P_train = split['train']['X'], split['train']['Y'], split['train']['P']
    target_train = (Y_train - pos_min) / (pos_max - pos_min)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs - config.warmup_epochs, eta_min=config.min_lr)
    criterion = nn.MSELoss()

    best_error, best_state, best_epoch, patience_counter = float('inf'), None, 0, 0
    for epoch in range(1, config.num_epochs + 1):
        if epoch <= config.warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.learning_rate * epoch / config.warmup_epochs

        model.train()
        perm = torch.randperm(len(X_train), device=X_train.device)
        total_loss = 0.0
        for i in range(0, len(perm), config.batch_size):
            idx = perm[i:i + config.batch_size]
            optimizer.zero_grad()
            # A new phase and timing reference is drawn every time a sample is used
            batch = randomize_reference(X_train[idx], True, True)
            loss = criterion(model(batch, None, P_train[idx]), target_train[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        if epoch > config.warmup_epochs:
            scheduler.step()

        val_error = distance_error(predict(model, split['val']), split['val']['Y'], pos_min, pos_max).mean().item()
        if val_error < best_error - config.min_delta:
            best_error, best_epoch, patience_counter = val_error, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        print(f"Epoch {epoch:3d}  train loss {total_loss / len(perm):.5f}  "
              f"val error {val_error:.3f} m  (best {best_error:.3f} m @ {best_epoch})", flush=True)

        if patience_counter >= config.patience:
            break

    model.load_state_dict(best_state)
    return best_error, best_epoch


def knn_predict(split, name, k):
    """Weighted k-nearest-neighbor regression on the CIRs of the training layouts."""
    X_train = split['train']['X'].flatten(1)
    dist = torch.cdist(split[name]['X'].flatten(1), X_train)
    d, idx = dist.topk(k, dim=1, largest=False)
    w = 1.0 / (d + 1e-6)
    return (split['train']['Y'][idx] * w.unsqueeze(-1)).sum(dim=1) / w.sum(dim=1, keepdim=True)


def summarize(errors, d):
    """Error statistics of one test set, overall and by the number of APs of the layout."""
    stats = {
        'mean_error': errors.mean().item(),
        'median_error': errors.median().item(),
        '90th_percentile': torch.quantile(errors, 0.9).item(),
        'num_samples': len(errors),
        'num_layouts': d['num_layouts'],
        'error_by_num_aps': {str(int(n)): errors[d['num_aps'] == n].mean().item()
                             for n in torch.unique(d['num_aps'])},
        'error_by_num_detected': {str(int(n)): errors[d['num_detected'] == n].mean().item()
                                  for n in torch.unique(d['num_detected'])}
    }
    return stats


def main():
    """Main experiment function."""
    args = parse_args()
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    config = get_config('regression')
    for key in ['num_layers', 'pooling', 'warmup_epochs', 'patience']:
        if getattr(args, key) is not None:
            setattr(config, key, getattr(args, key))
    config.use_anchor_position = args.use_anchor_position
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.lr is not None:
        config.learning_rate = args.lr
    config.seed = args.seed

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if args.device is not None
                          else 'cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    pos_min = torch.tensor(config.pos_min)
    pos_max = torch.tensor(config.pos_max)

    split = load_layout_split(args.data_dir, args.data_prefix, args.seed, pos_min, pos_max)
    if args.smoke:
        # Keep a few samples of every set and train for one epoch
        split = {name: {k: (v[:128] if torch.is_tensor(v) else v) for k, v in d.items()}
                 for name, d in split.items()}
        config.num_epochs, config.warmup_epochs = 1, 1
    split = {name: {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in d.items()}
             for name, d in split.items()}
    pos_min, pos_max = pos_min.to(device), pos_max.to(device)

    # Validation and test sets get one fixed random reference per AP and sample.
    # The training set gets a new one every time a sample is used (see train),
    # except for kNN, which has no training loop
    generator = torch.Generator(device=device).manual_seed(12345)
    for name in ['val', 'test_unseen', 'test_seen'] + (['train'] if args.model == 'knn' else []):
        split[name]['X'] = randomize_reference(split[name]['X'], True, True, generator)

    config.num_anchors = split['train']['X'].shape[1]
    config.input_shape = tuple(split['train']['X'].shape[2:])
    # The fixed layout of the configuration is not used: positions are given per sample
    config.anchor_positions = [[0.0, 0.0, 0.0]] * config.num_anchors
    print(f"{args.name}: train {len(split['train']['X'])} samples in {split['train']['num_layouts']} layouts, "
          f"val {len(split['val']['X'])} in {split['val']['num_layouts']}, "
          f"unseen test {len(split['test_unseen']['X'])} in {split['test_unseen']['num_layouts']}, "
          f"seen test {len(split['test_seen']['X'])} in {split['test_seen']['num_layouts']}, "
          f"up to {config.num_anchors} APs")

    result = {'name': args.name, 'model': args.model, 'seed': args.seed,
              'use_anchor_position': args.use_anchor_position,
              'num_train': len(split['train']['X']), 'num_val': len(split['val']['X'])}
    start = time.time()
    test_sets = ['test_unseen', 'test_seen']

    if args.model == 'knn':
        val_errors = {k: torch.sqrt(((knn_predict(split, 'val', k) - split['val']['Y']) ** 2).sum(-1)).mean().item()
                      for k in [1, 3, 5, 7, 9]}
        best_k = min(val_errors, key=val_errors.get)
        errors = {name: torch.sqrt(((knn_predict(split, name, best_k) - split[name]['Y']) ** 2).sum(-1))
                  for name in test_sets}
        result.update({'val_error': val_errors[best_k], 'k': best_k, 'num_params': 0})
    else:
        model = build_model(args, config).to(device)
        result['num_params'] = sum(p.numel() for p in model.parameters())
        if args.model == 'transformer':
            # The unused classification head is not part of the regression model
            result['num_params'] -= sum(p.numel() for p in model.classification_head.parameters())
        val_error, best_epoch = train(model, split, config, pos_min, pos_max)
        errors = {name: distance_error(predict(model, split[name]), split[name]['Y'], pos_min, pos_max)
                  for name in test_sets}
        result.update({'val_error': val_error, 'best_epoch': best_epoch})

    for name in test_sets:
        result[name] = summarize(errors[name], split[name])
    result['train_time_s'] = time.time() - start

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, f"{args.name}_seed{args.seed}")
    with open(path + '.json', 'w') as f:
        json.dump(result, f, indent=2)
    np.savez(path + '.npz', **{f'errors_{name}': errors[name].cpu().numpy() for name in test_sets},
             **{f'num_aps_{name}': split[name]['num_aps'].cpu().numpy() for name in test_sets})

    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
