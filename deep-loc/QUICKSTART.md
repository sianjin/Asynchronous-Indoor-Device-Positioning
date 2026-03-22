# Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Test

Test that everything is working:

```bash
# Test model instantiation
python3 -c "
from config.task_configs import RegressionConfig
from src.models.localization_model import IndoorLocalizationModel
import torch

config = RegressionConfig()
model = IndoorLocalizationModel(config)
print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
print('✓ Model created successfully!')
"
```

## Train Your First Model

### 1. Regression (3D Position Estimation)

Train a regression model for 3D position estimation:

```bash
python scripts/train.py --task regression --epochs 200 --batch_size 32
```

This will:
- Load data from [data.mat](data.mat)
- Train for 200 epochs
- Save checkpoints to `./checkpoints/`
- Save logs to `./logs/`

### 2. Monitor Training

Training logs are saved in real-time:

```bash
# View training log
tail -f logs/regression_baseline_*.log
```

### 3. Evaluate Trained Model

After training completes, evaluate the best model:

```bash
python scripts/evaluate.py --checkpoint checkpoints/regression_baseline_best.pth
```

This will show:
- Mean 3D error (meters)
- Median error
- Percentile errors (90th, 95th)
- Per-axis errors (x, y, z)

### 4. Run Inference

To run inference on new data:

```bash
# Create sample input (for testing)
python3 -c "
import numpy as np
# Create random sample with shape (4, 48, 32) = (anchors, taps, channels)
sample = np.random.randn(4, 48, 32).astype(np.float32)
np.save('sample_input.npy', sample)
print('Created sample_input.npy')
"

# Run inference
python scripts/inference.py \
    --checkpoint checkpoints/regression_baseline_best.pth \
    --input sample_input.npy
```

## Expected Results

With the default configuration:

- **Training time**: 1-2 hours on CPU, 30-60 minutes on GPU
- **Model size**: ~1.7M parameters
- **Expected performance**: 0.3-0.8 meters mean error
- **Target performance**: < 1.0 meter mean error

## Switching to Classification

To train a classification model instead:

```bash
python scripts/train.py --task classification --epochs 150
```

Note: Classification labels need to be properly parsed from the data file first.

## Common Commands

### Training with Custom Settings

```bash
# Train with different hyperparameters
python scripts/train.py \
    --task regression \
    --epochs 300 \
    --batch_size 64 \
    --lr 0.0005 \
    --embed_dim 512 \
    --num_layers 3 \
    --experiment_name my_experiment
```

### Resume Training from Checkpoint

```bash
python scripts/train.py \
    --task regression \
    --resume checkpoints/regression_baseline_epoch_50.pth
```

### Evaluate on Custom Data

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/regression_baseline_best.pth \
    --data_path /path/to/custom_data.mat
```

## Output Files

After training, you'll find:

```
checkpoints/
├── regression_baseline_best.pth          # Best model
├── regression_baseline_epoch_10.pth      # Periodic checkpoints
├── regression_baseline_epoch_20.pth
└── ...

logs/
├── regression_baseline_20260321_123456.log    # Training log
└── regression_baseline_metrics.json           # Metrics history
```

## Visualization

To visualize results, use the utilities in [src/utils/visualization.py](src/utils/visualization.py):

```python
import torch
import numpy as np
from src.utils.visualization import (
    plot_predictions_vs_ground_truth,
    plot_error_distribution,
    plot_3d_positions
)

# Load predictions and ground truth
# ... your code here ...

# Generate plots
plot_predictions_vs_ground_truth(predictions, ground_truth, save_path='predictions.png')
plot_error_distribution(predictions, ground_truth, save_path='errors.png')
plot_3d_positions(predictions, ground_truth, save_path='3d_positions.png')
```

## Troubleshooting

### Import Errors

If you get import errors, make sure you're running from the project root:

```bash
cd /path/to/deep-loc
python scripts/train.py ...
```

### CUDA Out of Memory

Reduce batch size or model size:

```bash
python scripts/train.py --batch_size 16 --embed_dim 128
```

### No CUDA Available

The code automatically falls back to CPU:

```bash
python scripts/train.py --device cpu
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore the [paper.pdf](paper.pdf) for the theoretical background
- Check the [config](config/) directory to customize hyperparameters
- Look at model architecture in [src/models/](src/models/)

## Support

For issues or questions:
1. Check the [README.md](README.md)
2. Review the code documentation
3. Open an issue in the repository
