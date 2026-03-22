"""
Logging utilities for training.
"""

import os
import json
from datetime import datetime


class Logger:
    """Simple logger for training progress."""

    def __init__(self, log_dir, experiment_name):
        """
        Initialize logger.

        Args:
            log_dir: directory to save logs
            experiment_name: name of the experiment
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # Create log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")

        # Initialize metrics history
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': []
        }

        self.log(f"Logging to: {self.log_file}")
        self.log(f"Experiment: {experiment_name}")
        self.log(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("-" * 80)

    def log(self, message, print_console=True):
        """
        Log a message.

        Args:
            message: message to log
            print_console: whether to print to console
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"

        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')

        # Print to console
        if print_console:
            print(log_message)

    def log_epoch(self, epoch, train_loss, val_loss, val_metrics, lr):
        """
        Log epoch results.

        Args:
            epoch: current epoch
            train_loss: training loss
            val_loss: validation loss
            val_metrics: dict of validation metrics
            lr: current learning rate
        """
        # Update history
        self.metrics_history['epochs'].append(epoch)
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['val_loss'].append(val_loss)

        # Add metrics to history
        for key, value in val_metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)

        # Log to file and console
        self.log("-" * 80)
        self.log(f"Epoch {epoch}")
        self.log(f"  Learning Rate: {lr:.6f}")
        self.log(f"  Train Loss: {train_loss:.6f}")
        self.log(f"  Val Loss: {val_loss:.6f}")

        # Log metrics
        for key, value in val_metrics.items():
            self.log(f"  {key}: {value:.6f}")

    def save_metrics(self):
        """Save metrics history to JSON file."""
        metrics_file = os.path.join(self.log_dir, f"{self.experiment_name}_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        self.log(f"Metrics saved to: {metrics_file}", print_console=False)

    def log_config(self, config):
        """
        Log configuration.

        Args:
            config: configuration object
        """
        self.log("Configuration:")
        config_dict = config.to_dict() if hasattr(config, 'to_dict') else vars(config)
        for key, value in sorted(config_dict.items()):
            if not key.startswith('_'):
                self.log(f"  {key}: {value}")
        self.log("-" * 80)


if __name__ == '__main__':
    # Test logger
    logger = Logger(log_dir='./logs', experiment_name='test')

    logger.log("This is a test message")

    # Test epoch logging
    val_metrics = {
        'mean_error': 0.5,
        'median_error': 0.4,
        'mae_x': 0.3,
        'mae_y': 0.4,
        'mae_z': 0.1
    }

    logger.log_epoch(
        epoch=1,
        train_loss=1.5,
        val_loss=1.2,
        val_metrics=val_metrics,
        lr=0.001
    )

    logger.save_metrics()
    print("\nLogger test passed!")
