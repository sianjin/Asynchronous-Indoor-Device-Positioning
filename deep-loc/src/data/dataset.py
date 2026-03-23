"""
Dataset class for indoor localization.
Loads and preprocesses data from .mat file.
"""

import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader


class IndoorLocalizationDataset(Dataset):
    """
    Dataset for indoor localization with Wi-Fi channel measurements.

    Data format:
    - Input: (Na, Ntap, 2M) per sample, where Na=4 anchors, Ntap=48 taps, 2M=32 channels
    - Regression output: (3,) - (x, y, z) position in meters
    - Classification output: (1,) - class label
    """

    def __init__(self, features, labels, task='regression', normalize_positions=True,
                 pos_min=None, pos_max=None):
        """
        Initialize dataset.

        Args:
            features: numpy array of shape (Ntap, 2M, Na, N_samples)
            labels: dict with 'regression' and/or 'classification' keys
            task: 'regression' or 'classification'
            normalize_positions: whether to normalize positions to [0, 1]
            pos_min: minimum position values [x_min, y_min, z_min]
            pos_max: maximum position values [x_max, y_max, z_max]
        """
        self.task = task
        self.normalize_positions = normalize_positions

        # Reshape features from (Ntap, 2M, Na, N) to (N, Na, Ntap, 2M)
        # This puts batch dimension first and makes anchors the second dimension
        self.features = np.transpose(features, (3, 2, 0, 1))  # (N, Na, Ntap, 2M)

        # Extract labels based on task
        if task == 'regression':
            # Regression labels: (3, N) -> (N, 3)
            self.labels = labels.T  # Now shape (N, 3)
        elif task == 'classification':
            # Classification labels: (N, 1) or (N,) with values 1-7 (MATLAB indexing)
            # Convert to 0-indexed for Python (0-6)
            if labels.ndim == 2:
                self.labels = labels.flatten() - 1  # Convert 1-7 to 0-6
            else:
                self.labels = labels - 1
            self.labels = self.labels.astype(np.int64)
        else:
            raise ValueError(f"Unknown task: {task}")

        # Position normalization parameters
        if normalize_positions and task == 'regression':
            if pos_min is None or pos_max is None:
                # Compute from data if not provided
                self.pos_min = self.labels.min(axis=0)
                self.pos_max = self.labels.max(axis=0)
            else:
                self.pos_min = np.array(pos_min, dtype=np.float32)
                self.pos_max = np.array(pos_max, dtype=np.float32)

            # Normalize labels
            self.labels = self._normalize_position(self.labels)
        else:
            self.pos_min = None
            self.pos_max = None

    def _normalize_position(self, pos):
        """Normalize position to [0, 1]."""
        return (pos - self.pos_min) / (self.pos_max - self.pos_min)

    def denormalize_position(self, pos_norm):
        """Denormalize position from [0, 1] to original range."""
        if self.pos_min is None or self.pos_max is None:
            return pos_norm
        return pos_norm * (self.pos_max - self.pos_min) + self.pos_min

    def __len__(self):
        """Return number of samples."""
        return len(self.features)

    def __getitem__(self, idx):
        """
        Get a single sample.

        Returns:
            features: (Na, Ntap, 2M) tensor
            label: (3,) tensor for regression or scalar for classification
        """
        features = torch.from_numpy(self.features[idx]).float()  # (Na, Ntap, 2M)
        label = torch.from_numpy(self.labels[idx]).float() if self.task == 'regression' \
            else torch.tensor(self.labels[idx], dtype=torch.long)

        return features, label


def load_data(data_path, task='regression', normalize_positions=True,
              pos_min=None, pos_max=None):
    """
    Load data from .mat file and create train/val datasets.

    Args:
        data_path: path to data.mat file
        task: 'regression' or 'classification'
        normalize_positions: whether to normalize positions
        pos_min: minimum position values
        pos_max: maximum position values

    Returns:
        train_dataset, val_dataset
    """
    # Load .mat file
    data = scipy.io.loadmat(data_path)

    # Extract training data
    train_X = data['training'][0, 0]['X']  # (48, 32, 4, 1152)
    train_Y = data['training'][0, 0]['Y']

    # Extract validation data
    val_X = data['validation'][0, 0]['X']  # (48, 32, 4, 288)
    val_Y = data['validation'][0, 0]['Y']

    # Extract labels based on task
    if task == 'regression':
        train_labels = train_Y[0, 0]['regression']  # (3, 1152)
        val_labels = val_Y[0, 0]['regression']  # (3, 288)
    elif task == 'classification':
        # TODO: Parse classification labels properly
        # For now, use placeholder
        train_labels = train_Y[0, 0]['classification']
        val_labels = val_Y[0, 0]['classification']
    else:
        raise ValueError(f"Unknown task: {task}")

    # Create datasets
    train_dataset = IndoorLocalizationDataset(
        train_X, train_labels, task=task,
        normalize_positions=normalize_positions,
        pos_min=pos_min, pos_max=pos_max
    )

    val_dataset = IndoorLocalizationDataset(
        val_X, val_labels, task=task,
        normalize_positions=normalize_positions,
        pos_min=train_dataset.pos_min if normalize_positions else None,
        pos_max=train_dataset.pos_max if normalize_positions else None
    )

    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, batch_size=32, num_workers=4,
                       pin_memory=True):
    """
    Create DataLoader objects for training and validation.

    Args:
        train_dataset: training dataset
        val_dataset: validation dataset
        batch_size: batch size for training
        num_workers: number of workers for data loading
        pin_memory: whether to pin memory

    Returns:
        train_loader, val_loader
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader
