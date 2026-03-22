"""
Trainer class for training and validation loops.
"""

import os
import torch
import torch.nn as nn
from tqdm import tqdm

from src.training.losses import build_loss_fn
from src.training.metrics import compute_metrics


class Trainer:
    """Training manager for localization model."""

    def __init__(self, model, train_loader, val_loader, config, logger):
        """
        Initialize trainer.

        Args:
            model: localization model
            train_loader: training data loader
            val_loader: validation data loader
            config: configuration object
            logger: logger object
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger

        # Setup device
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available() and config.device == 'cuda':
            self.logger.log("CUDA not available, using CPU instead")
            self.device = torch.device('cpu')

        self.model.to(self.device)

        # Setup loss function
        self.criterion = build_loss_fn(config)

        # Setup optimizer
        self.optimizer = self._build_optimizer()

        # Setup scheduler
        self.scheduler = self._build_scheduler()

        # Training state
        self.current_epoch = 0
        self.best_metric = float('inf') if config.task == 'regression' else 0.0
        self.patience_counter = 0

        # Create checkpoint directory
        os.makedirs(config.save_dir, exist_ok=True)

    def _build_optimizer(self):
        """Build optimizer based on config."""
        if self.config.optimizer == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _build_scheduler(self):
        """Build learning rate scheduler based on config."""
        if self.config.lr_scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs - self.config.warmup_epochs,
                eta_min=self.config.min_lr
            )
        elif self.config.lr_scheduler == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.step_size,
                gamma=self.config.gamma
            )
        elif self.config.lr_scheduler == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min' if self.config.task == 'regression' else 'max',
                patience=self.config.patience_scheduler,
                factor=0.5
            )
        elif self.config.lr_scheduler == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler: {self.config.lr_scheduler}")

    def train_epoch(self, epoch):
        """
        Train for one epoch.

        Args:
            epoch: current epoch number

        Returns:
            average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.num_epochs}")

        for batch_idx, (data, target) in enumerate(pbar):
            # Move data to device
            data = data.to(self.device)
            target = target.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(data)

            # Compute loss
            loss = self.criterion(output, target)

            # Backward pass
            loss.backward()

            # Update weights
            self.optimizer.step()

            # Update statistics
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            # Log interval
            if batch_idx % self.config.log_interval == 0 and batch_idx > 0:
                avg_loss = total_loss / num_batches
                self.logger.log(
                    f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {avg_loss:.6f}",
                    print_console=False
                )

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self):
        """
        Validate on validation set.

        Returns:
            average validation loss, dict of metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validating"):
                # Move data to device
                data = data.to(self.device)
                target = target.to(self.device)

                # Forward pass
                output = self.model(data)

                # Compute loss
                loss = self.criterion(output, target)
                total_loss += loss.item()

                # Store predictions and targets
                all_preds.append(output.cpu())
                all_targets.append(target.cpu())

        # Concatenate all predictions and targets
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Compute metrics
        metrics = compute_metrics(all_preds, all_targets, task=self.config.task)

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss, metrics

    def train(self):
        """Main training loop."""
        self.logger.log("Starting training...")
        self.logger.log(f"Device: {self.device}")
        self.logger.log(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.log("-" * 80)

        for epoch in range(1, self.config.num_epochs + 1):
            self.current_epoch = epoch

            # Warmup learning rate
            if epoch <= self.config.warmup_epochs:
                lr = self.config.learning_rate * epoch / self.config.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

            # Train one epoch
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss, val_metrics = self.validate()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log epoch results
            self.logger.log_epoch(epoch, train_loss, val_loss, val_metrics, current_lr)

            # Update scheduler
            if self.scheduler is not None and epoch > self.config.warmup_epochs:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Check for best model
            current_metric = val_metrics.get('mean_error', val_loss) if self.config.task == 'regression' \
                else val_metrics.get('accuracy', -val_loss)

            is_best = (current_metric < self.best_metric) if self.config.task == 'regression' \
                else (current_metric > self.best_metric)

            if is_best:
                self.best_metric = current_metric
                self.patience_counter = 0
                self._save_checkpoint(epoch, is_best=True)
                self.logger.log(f"New best model saved! Metric: {current_metric:.6f}")
            else:
                self.patience_counter += 1

            # Save latest checkpoint
            if epoch % 10 == 0:
                self._save_checkpoint(epoch, is_best=False)

            # Early stopping
            if self.config.early_stopping and self.patience_counter >= self.config.patience:
                self.logger.log(f"Early stopping triggered after {epoch} epochs")
                break

        # Save final metrics
        self.logger.save_metrics()
        self.logger.log("Training completed!")

    def _save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else vars(self.config)
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if is_best:
            path = os.path.join(self.config.save_dir, f"{self.config.experiment_name}_best.pth")
        else:
            path = os.path.join(self.config.save_dir, f"{self.config.experiment_name}_epoch_{epoch}.pth")

        torch.save(checkpoint, path)
        self.logger.log(f"Checkpoint saved: {path}", print_console=False)

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']

        self.logger.log(f"Checkpoint loaded: {checkpoint_path}")
        self.logger.log(f"Resuming from epoch {self.current_epoch}")
