"""
Experiment runner for the model comparison in the paper.

Trains one model on a position-disjoint train/validation split and evaluates
the best-validation checkpoint on separately generated test positions.
All models share the same split, optimizer, schedule and stopping rule.
"""

import os
import sys
import json
import time
import argparse
import itertools
import numpy as np
import scipy.io
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.task_configs import get_config
from src.models.localization_model import IndoorLocalizationModel
from src.models.baseline_cnn import EarlyFusionCNN


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run one experiment of the model comparison')

    parser.add_argument('--name', type=str, required=True, help='Experiment name')
    parser.add_argument('--model', type=str, default='transformer',
                       choices=['transformer', 'cnn', 'knn'], help='Model type')
    parser.add_argument('--input', type=str, default='complex',
                       choices=['complex', 'magnitude'], help='Input representation')
    parser.add_argument('--seed', type=int, default=0, help='Seed for the split and the training')
    parser.add_argument('--data_dir', type=str, default='.',
                       help='Directory with the data_*.mat files')
    parser.add_argument('--train_condition', type=str, default='nominal',
                       help='Impairment condition of the training set')
    parser.add_argument('--output_dir', type=str, default='results/camera_ready',
                       help='Directory to save results')

    # Model variants (transformer only)
    parser.add_argument('--num_layers', type=int, default=None,
                       help='Number of transformer layers (0 = no cross-anchor attention)')
    parser.add_argument('--pooling', type=str, default=None, choices=['mean', 'attention'])
    parser.add_argument('--anchor_dropout', type=float, default=None)
    parser.add_argument('--use_anchor_position', action='store_true')

    # Training
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--warmup_epochs', type=int, default=None)
    parser.add_argument('--patience', type=int, default=None, help='Early stopping patience in epochs')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'mps', 'cpu'],
                       help='Device to use (default: cuda if available, otherwise cpu)')
    parser.add_argument('--threads', type=int, default=None, help='Number of CPU threads')
    parser.add_argument('--smoke', action='store_true',
                       help='Quick test of the pipeline: one epoch on a small subset of the data')

    return parser.parse_args()


def load_data_file(path, input_mode):
    """Load one data_*.mat file written by phy/wifiPosGenerateData.m."""
    data = scipy.io.loadmat(path)
    X = np.transpose(data['X'], (3, 2, 0, 1)).astype(np.float32)  # (N, Na, Ntap, 2M)
    if input_mode == 'magnitude':
        M = X.shape[-1] // 2
        X = np.sqrt(X[..., :M] ** 2 + X[..., M:] ** 2)  # (N, Na, Ntap, M)
    return {
        'X': X,
        'Y': data['position'].T.astype(np.float32),  # (N, 3)
        'snr': data['snr'].reshape(-1),  # (N,)
        'los': data['los'].T.astype(bool),  # (N, Na)
        'anchor_positions': data['apPositions'].T.astype(np.float32)  # (Na, 3)
    }


def load_split(data_dir, train_condition, seed, input_mode, val_frac=0.15):
    """
    Load the datasets written by phy/wifiPosGenerateData.m.

    The training file is split by device position into training and validation
    sets. The test positions come from separate files, one per impairment
    condition, and are never used for training or model selection.

    Returns:
        split: dict with 'train', 'val', 'test' -> (features, positions), where
            'test' has the same impairment condition as the training set
        extra: dict with the per-sample 'snr' and 'num_los' of the test set, the
            test sets of the other conditions and the AP coordinates
    """
    train = load_data_file(os.path.join(data_dir, f'data_train_{train_condition}.mat'), input_mode)

    # Samples in which no AP was detected carry no information about the position
    def detected(d):
        return np.abs(d['X']).sum(axis=(1, 2, 3)) > 0

    _, group = np.unique(np.round(train['Y'], 3), axis=0, return_inverse=True)
    group = group.reshape(-1)
    num_groups = group.max() + 1
    order = np.random.RandomState(seed).permutation(num_groups)
    val_groups = order[:int(round(val_frac * num_groups))]
    is_val = np.isin(group, val_groups)
    keep = detected(train)

    train_idx = keep & ~is_val
    X_train = train['X'][train_idx]
    scale = X_train[np.abs(X_train).sum(axis=(2, 3)) > 0].std()

    def to_set(d, idx):
        return torch.from_numpy(d['X'][idx] / scale), torch.from_numpy(d['Y'][idx])

    split = {'train': to_set(train, train_idx), 'val': to_set(train, keep & is_val)}

    extra = {'conditions': {}, 'anchor_positions': train['anchor_positions'].tolist()}
    prefix = os.path.join(data_dir, 'data_test_')
    for path in sorted(p for p in os.listdir(data_dir) if p.startswith('data_test_')):
        condition = path[len('data_test_'):-len('.mat')]
        test = load_data_file(prefix + condition + '.mat', input_mode)
        idx = detected(test)
        extra['conditions'][condition] = to_set(test, idx)
        if condition == train_condition:
            split['test'] = extra['conditions'][condition]
            extra['snr'] = test['snr'][idx]
            extra['num_los'] = test['los'][idx].sum(axis=1)
    return split, extra


def build_model(args, config):
    """Build the model for this experiment."""
    if args.model == 'cnn':
        return EarlyFusionCNN(input_shape=config.input_shape, num_anchors=config.num_anchors)
    return IndoorLocalizationModel(config)


def predict(model, X, anchor_mask=None, batch_size=256):
    """Predict normalized positions for all samples."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            mask = None if anchor_mask is None else anchor_mask[i:i + batch_size]
            preds.append(model(X[i:i + batch_size], mask))
    return torch.cat(preds, dim=0)


def distance_error(pred_norm, target, pos_min, pos_max):
    """3D distance error in meters from normalized predictions."""
    pred = pred_norm * (pos_max - pos_min) + pos_min
    return torch.sqrt(((pred - target) ** 2).sum(dim=-1))


def train(model, split, config, pos_min, pos_max):
    """Train with warmup + cosine schedule and early stopping on the validation error."""
    X_train, Y_train = split['train']
    X_val, Y_val = split['val']
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
            loss = criterion(model(X_train[idx]), target_train[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        if epoch > config.warmup_epochs:
            scheduler.step()

        val_error = distance_error(predict(model, X_val), Y_val, pos_min, pos_max).mean().item()
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


def error_by_num_anchors(model, split, pos_min, pos_max):
    """
    Test error as a function of the number of available APs.

    Every subset of the AP slots is evaluated with the other APs removed
    (zeroed and masked). Errors are grouped by the number of detected APs
    that remain in the subset.
    """
    X, Y = split['test']
    Na = X.shape[1]
    detected = X.abs().sum(dim=(2, 3)) > 0  # (N, Na)

    errors = {n: [] for n in range(1, Na + 1)}
    for k in range(1, Na + 1):
        for subset in itertools.combinations(range(Na), k):
            removed = torch.ones(len(X), Na, dtype=torch.bool, device=X.device)
            removed[:, list(subset)] = False
            X_subset = X.masked_fill(removed[:, :, None, None], 0.0)
            err = distance_error(predict(model, X_subset, removed), Y, pos_min, pos_max)
            num_available = (detected & ~removed).sum(dim=1)
            for n in range(1, Na + 1):
                errors[n].append(err[num_available == n])

    return {n: torch.cat(e).mean().item() for n, e in errors.items()}


def run_knn(split, pos_min, pos_max):
    """Weighted k-nearest-neighbor fingerprinting; k is selected on the validation set."""
    X_train, Y_train = split['train']
    X_train = X_train.flatten(1)

    def knn_predict(X, k):
        dist = torch.cdist(X.flatten(1), X_train)
        d, idx = dist.topk(k, dim=1, largest=False)
        w = 1.0 / (d + 1e-6)
        return (Y_train[idx] * w.unsqueeze(-1)).sum(dim=1) / w.sum(dim=1, keepdim=True)

    X_val, Y_val = split['val']
    val_errors = {k: torch.sqrt(((knn_predict(X_val, k) - Y_val) ** 2).sum(-1)).mean().item()
                  for k in [1, 3, 5, 7, 9]}
    best_k = min(val_errors, key=val_errors.get)

    X_test, Y_test = split['test']
    pred = knn_predict(X_test, best_k)
    return pred, val_errors[best_k], best_k


def main():
    """Main experiment function."""
    args = parse_args()
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    config = get_config('regression')
    for key in ['num_layers', 'pooling', 'anchor_dropout', 'warmup_epochs', 'patience']:
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

    split, extra = load_split(args.data_dir, args.train_condition, args.seed, args.input)
    if args.smoke:
        # Keep a few samples of every set and train for one epoch
        num_smoke = 128
        split = {name: (X[:num_smoke], Y[:num_smoke]) for name, (X, Y) in split.items()}
        extra['conditions'] = {name: (X[:num_smoke], Y[:num_smoke])
                               for name, (X, Y) in extra['conditions'].items()}
        extra['snr'], extra['num_los'] = extra['snr'][:num_smoke], extra['num_los'][:num_smoke]
        config.num_epochs, config.warmup_epochs = 1, 1
    config.anchor_positions = extra['anchor_positions']
    config.num_anchors = len(extra['anchor_positions'])
    device = torch.device(args.device if args.device is not None
                          else 'cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        # All inputs have the same size, so let cuDNN pick the fastest convolution algorithm
        torch.backends.cudnn.benchmark = True
    split = {name: (X.to(device), Y.to(device)) for name, (X, Y) in split.items()}
    extra['conditions'] = {name: (X.to(device), Y.to(device))
                           for name, (X, Y) in extra['conditions'].items()}

    config.input_shape = tuple(split['train'][0].shape[2:])
    pos_min = torch.tensor(config.pos_min, device=device)
    pos_max = torch.tensor(config.pos_max, device=device)
    print(f"{args.name}: train {len(split['train'][0])}, val {len(split['val'][0])}, "
          f"test {len(split['test'][0])} samples, input {config.input_shape}")

    X_test, Y_test = split['test']
    result = {'name': args.name, 'model': args.model, 'input': args.input, 'seed': args.seed,
              'num_train': len(split['train'][0]), 'num_val': len(split['val'][0]),
              'num_test': len(X_test)}
    start = time.time()

    if args.model == 'knn':
        pred, val_error, best_k = run_knn(split, pos_min, pos_max)
        errors = torch.sqrt(((pred - Y_test) ** 2).sum(-1))
        result.update({'val_error': val_error, 'k': best_k, 'num_params': 0})
    else:
        model = build_model(args, config).to(device)
        result['num_params'] = sum(p.numel() for p in model.parameters())
        if args.model == 'transformer':
            # The unused classification head is not part of the regression model
            result['num_params'] -= sum(p.numel() for p in model.classification_head.parameters())
        val_error, best_epoch = train(model, split, config, pos_min, pos_max)
        pred_norm = predict(model, X_test)
        pred = pred_norm * (pos_max - pos_min) + pos_min
        errors = distance_error(pred_norm, Y_test, pos_min, pos_max)
        result.update({'val_error': val_error, 'best_epoch': best_epoch,
                       'error_by_num_anchors': error_by_num_anchors(model, split, pos_min, pos_max)})

    result.update({
        'test_mean_error': errors.mean().item(),
        'test_median_error': errors.median().item(),
        'test_90th_percentile': torch.quantile(errors, 0.9).item(),
        'train_time_s': time.time() - start
    })

    errors = errors.cpu()
    # Breakdown of the test error by SNR and by number of line-of-sight APs
    result['train_condition'] = args.train_condition
    result['error_by_snr'] = {str(int(v)): errors[torch.from_numpy(extra['snr'] == v)].mean().item()
                              for v in np.unique(extra['snr'])}
    result['error_by_num_los'] = {str(int(v)): errors[torch.from_numpy(extra['num_los'] == v)].mean().item()
                                  for v in np.unique(extra['num_los'])}
    # Test error under the other impairment conditions (same test positions)
    if args.model != 'knn':
        result['error_by_condition'] = {
            name: distance_error(predict(model, X), Y, pos_min, pos_max).mean().item()
            for name, (X, Y) in extra['conditions'].items()}

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, f"{args.name}_seed{args.seed}")
    with open(path + '.json', 'w') as f:
        json.dump(result, f, indent=2)
    np.savez(path + '.npz', errors=errors.numpy(), predicted_positions=pred.cpu().numpy(),
             ground_truth_positions=Y_test.cpu().numpy(),
             num_detected_anchors=(X_test.abs().sum(dim=(2, 3)) > 0).sum(dim=1).cpu().numpy(),
             snr=extra['snr'], num_los=extra['num_los'])

    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
