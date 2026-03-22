# Indoor Deep Localization with Wi-Fi

PyTorch implementation of transformer-based deep learning framework for Wi-Fi indoor localization, based on the paper "Wi-Fi Indoor Deep Device Track".

## Overview

This project implements a permutation-invariant neural network architecture for estimating 3D positions from Wi-Fi channel state information (CSI). The model uses:

- **Hybrid CNN + MLP Encoder**: Extracts features from per-anchor channel measurements
- **Cross-Anchor Transformer**: Fuses information across multiple access points via self-attention
- **Task-Specific Heads**: Supports both regression (3D position) and classification tasks

### Key Features

- Easy switching between regression and classification tasks via configuration
- Permutation-invariant architecture supporting variable number of anchors
- Modular codebase with clear separation of concerns
- Comprehensive training, evaluation, and inference scripts
- Detailed logging and checkpointing

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- SciPy >= 1.10.0

### Setup

```bash
# Clone or navigate to project directory
cd deep-loc

# Install dependencies
pip install -r requirements.txt
```

## Data Format

The model expects data in MATLAB `.mat` format with the following structure:

- **Features**: Shape `(Ntap, 2M, Na, Nsample)` where:
  - `Ntap = 48`: Number of delay taps (time-domain channel impulse response)
  - `2M = 32`: Real + imaginary parts of channel (M = Nr × Ns)
  - `Na = 4`: Number of anchors (access points)
  - `Nsample`: Number of samples (1152 training, 288 validation)

- **Regression Labels**: Shape `(3, Nsample)` - 3D positions (x, y, z) in meters
- **Classification Labels**: Structured array (future use)

## Usage

### Training

Train a regression model (3D position estimation):

```bash
python scripts/train.py --task regression --epochs 200 --batch_size 32
```

Train a classification model:

```bash
python scripts/train.py --task classification --epochs 150
```

### Training Options

```bash
python scripts/train.py \
    --task regression \              # Task: 'regression' or 'classification'
    --data_path data.mat \           # Path to data file
    --batch_size 32 \                # Batch size
    --epochs 200 \                   # Number of epochs
    --lr 0.001 \                     # Learning rate
    --weight_decay 0.0001 \          # Weight decay
    --embed_dim 256 \                # Embedding dimension
    --num_layers 2 \                 # Number of transformer layers
    --num_heads 4 \                  # Number of attention heads
    --device cuda \                  # Device: 'cuda' or 'cpu'
    --experiment_name my_exp \       # Experiment name
    --resume checkpoint.pth          # Resume from checkpoint (optional)
```

### Evaluation

Evaluate a trained model:

```bash
python scripts/evaluate.py --checkpoint checkpoints/regression_baseline_best.pth
```

### Inference

Run inference on new data:

```bash
# Input should be .npy file with shape (Na, Ntap, 2M) or (batch, Na, Ntap, 2M)
python scripts/inference.py \
    --checkpoint checkpoints/regression_baseline_best.pth \
    --input sample_data.npy
```

## Model Architecture

### 1. Per-Anchor Encoder (Hybrid CNN + MLP)

Each anchor's channel measurement `(48, 32)` is processed independently:

- **CNN Branch**: 3 convolutional layers to extract spatial-temporal patterns
- **MLP Branch**: Fully connected layers to capture global statistics
- **Fusion**: Concatenate features and project to embedding dimension

### 2. Cross-Anchor Transformer

Fuses information across anchors using self-attention:

- Multi-head self-attention (4 heads by default)
- Feed-forward network
- 2 transformer layers
- No positional encoding (permutation invariant)

### 3. Global Aggregation

Mean pooling across anchors to create fixed-dimensional representation

### 4. Task-Specific Head

- **Regression**: MLP outputting 3D position (x, y, z)
- **Classification**: MLP outputting class logits

## Configuration

Default hyperparameters are in [config/default_config.py](config/default_config.py):

```python
# Model Architecture
embed_dim = 256
cnn_channels = [32, 64, 128]
mlp_hidden_dim = 256
num_heads = 4
num_layers = 2
ff_dim = 512
dropout = 0.1

# Training
batch_size = 32
num_epochs = 200
learning_rate = 1e-3
weight_decay = 1e-4
optimizer = 'adamw'
lr_scheduler = 'cosine'
warmup_epochs = 10
```

Task-specific configs are in [config/task_configs.py](config/task_configs.py).

## Project Structure

```
deep-loc/
├── config/                      # Configuration files
│   ├── default_config.py        # Default hyperparameters
│   └── task_configs.py          # Task-specific configs
├── src/
│   ├── data/                    # Data loading and preprocessing
│   │   └── dataset.py
│   ├── models/                  # Model architecture
│   │   ├── encoder.py           # Hybrid CNN+MLP encoder
│   │   ├── transformer.py       # Cross-anchor transformer
│   │   ├── heads.py             # Task-specific heads
│   │   └── localization_model.py # Complete model
│   ├── training/                # Training infrastructure
│   │   ├── trainer.py           # Training loop
│   │   ├── losses.py            # Loss functions
│   │   └── metrics.py           # Evaluation metrics
│   └── utils/                   # Utilities
│       ├── logger.py            # Logging
│       └── visualization.py     # Plotting
├── scripts/                     # User-facing scripts
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   └── inference.py             # Inference script
├── data.mat                     # Training data
├── paper.pdf                    # Reference paper
└── requirements.txt             # Dependencies
```

## Metrics

### Regression Task

- **Mean Error**: Mean 3D Euclidean distance in meters
- **Median Error**: Median 3D Euclidean distance
- **Percentile Errors**: 90th and 95th percentile errors
- **Per-Axis MAE**: Mean absolute error for x, y, z
- **Per-Axis RMSE**: Root mean squared error for x, y, z

### Classification Task

- **Accuracy**: Top-1 classification accuracy
- **Top-5 Accuracy**: Top-5 classification accuracy

## Expected Performance

Based on the small dataset (1152 training samples):

- **Target**: < 1.0 meter mean error for 3D position estimation
- **Expected**: 0.3-0.8 meters (typical for Wi-Fi localization)
- **Room scale**: 5m × 8m × 1m

## Tips and Tricks

### Avoiding Overfitting

The dataset is small (1152 samples), so overfitting is a concern:

- Dropout is used throughout the model (0.1)
- Early stopping with patience=30
- Weight decay for regularization
- Consider reducing model size if overfitting occurs

### Hyperparameter Tuning

Key hyperparameters to tune:

1. **Learning rate**: Try 1e-4 to 1e-2
2. **Batch size**: 16, 32, or 64
3. **Embedding dimension**: 128, 256, or 512
4. **Number of layers**: 1, 2, or 3

### GPU vs CPU

- Training on GPU: ~30-60 minutes for 200 epochs
- Training on CPU: ~2-4 hours for 200 epochs
- Model has ~750K parameters

## Troubleshooting

### CUDA Out of Memory

- Reduce batch size: `--batch_size 16`
- Reduce embedding dimension: `--embed_dim 128`

### Poor Performance

- Check data loading is correct
- Verify normalization is applied
- Try different learning rates
- Increase number of epochs

### Classification Labels Not Working

The classification labels in the current data file need special parsing. Focus on regression task first.

## Citation

If you use this code, please cite the reference paper:

```
Wi-Fi Indoor Deep Device Track
```

## License

This project is for research purposes.

## Contact

For questions or issues, please open an issue in the repository.
