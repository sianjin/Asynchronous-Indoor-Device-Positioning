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

**For Regression:**
```bash
python scripts/evaluate.py --checkpoint checkpoints/regression_baseline_best.pth
```

**For Classification:**
```bash
python scripts/evaluate.py --checkpoint checkpoints/classification_baseline_best.pth
```

The script automatically detects the task type (regression or classification) from the checkpoint file and displays the appropriate metrics. See below for explanation of each metric.

### 4. Understanding Evaluation Metrics

#### Regression Metrics (3D Position Estimation)

**Primary Metrics** (denormalized - in meters):

| Metric | What It Means | Example |
|--------|--------------|---------|
| **`mean_error`** | Average 3D positioning error | `0.477m` = 47.7 cm average error |
| **`median_error`** | 50% of predictions better than this | `0.233m` = half within 23.3 cm |
| **`90th_percentile`** | 90% of predictions within this | `1.014m` = 90% within 1 meter |
| **`95th_percentile`** | 95% of predictions within this | `1.642m` = 95% within 1.64 m |

**Per-Axis Metrics:**

| Metric | What It Means | Example |
|--------|--------------|---------|
| **`mae_x/y/z`** | Average error per coordinate (MAE) | `0.200m` = ±20 cm in that axis |
| **`rmse_x/y/z`** | Error spread per coordinate (RMSE) | `0.491m` (penalizes large errors more) |

**Quick Interpretation:**
```
mean_error: 0.477m     → Average positioning error (PRIMARY METRIC)
median_error: 0.233m   → 50% of predictions within 23 cm
90th_percentile: 1.014m → 90% within 1 meter
mae_z: 0.033m          → Excellent height accuracy (3.3 cm)
```

**Performance Guide:**
- **< 0.5m**: Excellent indoor localization
- **0.5-1.0m**: Good performance
- **> 1.0m**: Needs improvement

---

#### Classification Metrics (Location Class Prediction)

**Metrics:**

| Metric | Formula | What It Means | Example |
|--------|---------|--------------|---------|
| **`accuracy`** | `correct / total` | % of correct predictions **(PRIMARY)** | `1.000000` = 100% perfect! |
| **`top5_accuracy`** | `true in top-5 / total` | % where true class in top-5 | `0.956` = 95.6% in top-5 |

**Quick Interpretation:**
```
accuracy: 1.000000     → 100% correct (perfect classification!)
top5_accuracy: 1.000000 → True class always in top-5
```

**Performance Guide:**
- **100%**: Perfect classification
- **> 90%**: Excellent
- **80-90%**: Good
- **70-80%**: Fair
- **< 70%**: Needs improvement

**Note**: `top5_accuracy` ≥ `accuracy` (always)

---

#### Task Comparison

| | Regression | Classification |
|---|------------|----------------|
| **Output** | Continuous 3D coordinates (x, y, z) | Discrete location class (0-99) |
| **Primary Metric** | `mean_error` (meters) | `accuracy` (%) |
| **Use Case** | Precise positioning | Room/zone identification |

### 5. Run Inference

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
- **Expected performance** (denormalized metrics):
  - `mean_error`: 0.3-0.8 meters (average positioning error)
  - `median_error`: 0.2-0.5 meters (median positioning error)
  - `90th_percentile`: < 1.5 meters (90% of predictions within this)
  - Per-axis MAE: 0.1-0.5 meters depending on axis
- **Target performance**: `mean_error` < 1.0 meter

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
