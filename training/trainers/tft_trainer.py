"""
TFT-Lite Trainer with mixed precision and GPU optimization.

Design Decisions:

1. Mixed Precision (AMP):
   - RTX 4090 Tensor Cores optimized for FP16/BF16
   - ~2x memory reduction, ~1.5x speedup
   - GradScaler for numerical stability

2. Gradient Handling:
   - Gradient clipping prevents attention instabilities
   - Gradient accumulation for effective larger batch sizes

3. Optimizer:
   - AdamW with weight decay (regularization)
   - Cosine annealing scheduler
   - Warmup period for stability

4. Evaluation:
   - PhysioNet Clinical Utility Score
   - Patient-level predictions for proper utility calculation
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from training.config.tft_config import TFTConfig
from training.trainers.base_trainer import BaseTrainer
from training.data.dataset import (
    SepsisSequenceDataset,
    create_data_loaders,
    collate_sequences,
)
from training.data.cross_validation import get_patient_file_mapping

# Import model architecture
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from models.tft_lite.architecture import TFTLite

logger = logging.getLogger(__name__)


class TFTTrainer(BaseTrainer):
    """
    TFT-Lite trainer with mixed precision and GPU optimization.

    Features:
    - Automatic Mixed Precision (AMP) for RTX 4090
    - Gradient clipping for attention stability
    - Patient-level sequence modeling
    - Clinical Utility Score optimization

    Example:
        >>> config = TFTConfig(data_dir="data/physionet", device="cuda")
        >>> trainer = TFTTrainer(config)
        >>> results = trainer.train()
        >>> best_model = trainer.get_best_model()
    """

    def __init__(
        self,
        config: TFTConfig,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize TFT trainer.

        Args:
            config: TFT training configuration
            output_dir: Output directory (overrides config)
        """
        super().__init__(config, output_dir)
        self.config: TFTConfig = config

        # Device setup
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"
        )
        if self.device.type == "cuda":
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.warning("CUDA not available, using CPU")

        # Mixed precision
        self.scaler = GradScaler(enabled=config.use_amp)
        self.amp_dtype = torch.float16 if config.amp_dtype == "float16" else torch.bfloat16

        # Normalization stats
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    def _load_data(
        self,
        train_patients: List[str],
        val_patients: List[str],
    ) -> Tuple[Any, Any]:
        """
        Load and preprocess data for TFT.

        Args:
            train_patients: Training patient IDs
            val_patients: Validation patient IDs

        Returns:
            Tuple of (train_loader, val_loader)
        """
        logger.info("Creating data loaders...")

        train_loader, val_loader = create_data_loaders(
            train_patients=train_patients,
            val_patients=val_patients,
            data_dir=self.config.data_dir,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            max_seq_len=self.config.max_seq_len,
            for_tft=True,
        )

        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Val batches: {len(val_loader)}")
        logger.info(f"  Batch size: {self.config.batch_size}")

        return train_loader, val_loader

    def _create_model(self) -> TFTLite:
        """Create TFT-Lite model."""
        model_kwargs = self.config.get_model_kwargs()
        model = TFTLite(
            input_size=model_kwargs["input_size"],
            hidden_size=self.config.hidden_size,
            lstm_layers=self.config.n_encoder_layers,  # Map to LSTM layers
            attention_heads=self.config.n_heads,  # Map to attention heads
            output_size=1,  # Binary classification
            dropout=self.config.dropout,
            max_seq_length=self.config.max_seq_len,
        )
        model = model.to(self.device)

        # Log model info
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {n_params:,}")

        return model

    def _create_optimizer_scheduler(
        self,
        model: nn.Module,
        n_training_steps: int,
    ) -> Tuple[torch.optim.Optimizer, Any]:
        """Create optimizer and learning rate scheduler."""
        optimizer = AdamW(
            model.parameters(),
            **self.config.get_optimizer_kwargs(),
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            **self.config.get_scheduler_kwargs(n_training_steps),
        )

        return optimizer, scheduler

    def _train_fold(
        self,
        train_data: Any,  # DataLoader
        val_data: Any,  # DataLoader
        fold: int,
    ) -> Tuple[nn.Module, Dict[str, float]]:
        """
        Train TFT model on a single fold.

        Args:
            train_data: Training DataLoader
            val_data: Validation DataLoader
            fold: Fold index

        Returns:
            Tuple of (trained_model, metrics_dict)
        """
        logger.info(f"Training TFT fold {fold}...")

        # Create model
        model = self._create_model()

        # Compute class weight for loss
        if self.config.pos_weight is None:
            # Estimate from first batch
            batch = next(iter(train_data))
            labels = batch[1]
            n_pos = (labels == 1).sum().float()
            n_neg = (labels == 0).sum().float()
            pos_weight = n_neg / n_pos if n_pos > 0 else torch.tensor(14.0)
        else:
            pos_weight = torch.tensor(self.config.pos_weight)

        pos_weight = pos_weight.to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Optimizer and scheduler
        n_training_steps = len(train_data) * self.config.n_epochs
        optimizer, scheduler = self._create_optimizer_scheduler(model, n_training_steps)

        # Training loop
        best_metrics = {}
        best_model_state = None

        for epoch in range(self.config.n_epochs):
            # Train epoch
            train_loss = self._train_epoch(
                model, train_data, criterion, optimizer, scheduler, epoch
            )

            # Validate
            val_loss, metrics = self._validate(model, val_data, criterion)

            # Log
            self.logger.log_epoch(
                epoch=epoch,
                fold=fold,
                train_loss=train_loss,
                val_loss=val_loss,
                metrics=metrics,
                learning_rate=scheduler.get_last_lr()[0],
            )

            # Check early stopping
            primary_metric = metrics[self.config.primary_metric]
            if self.early_stopping(primary_metric, epoch):
                logger.info(f"Early stopping at epoch {epoch}")
                break

            # Save best model
            if self.checkpoint_manager.is_better(primary_metric):
                best_metrics = metrics.copy()
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                self.checkpoint_manager.save_pytorch(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    fold=fold,
                    metric_value=primary_metric,
                    config=self.config.to_dict(),
                    scheduler=scheduler,
                )

        # Restore best model
        if best_model_state:
            model.load_state_dict(best_model_state)

        logger.info(f"Fold {fold} complete: {best_metrics}")
        return model, best_metrics

    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: Any,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
    ) -> float:
        """Train a single epoch."""
        model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            features, labels, mask, _ = batch
            features = features.to(self.device).float()
            labels = labels.to(self.device).float()
            mask = mask.to(self.device).float()

            # Get the last valid label for each sequence
            # TFTLite outputs prediction for last timestep only
            batch_size = features.size(0)
            last_labels = torch.zeros(batch_size, 1, device=self.device)
            for i in range(batch_size):
                valid_len = mask[i].sum().int().item()
                if valid_len > 0:
                    last_labels[i] = labels[i, valid_len - 1]

            optimizer.zero_grad()

            # Mixed precision forward pass
            with autocast(enabled=self.config.use_amp, dtype=self.amp_dtype):
                outputs = model(features)  # TFTLite: (batch, seq, features) -> (batch, 1)
                loss = criterion(outputs, last_labels)

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_val)

            # Optimizer step
            self.scaler.step(optimizer)
            self.scaler.update()

            scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / n_batches

    def _validate(
        self,
        model: nn.Module,
        val_loader: Any,
        criterion: nn.Module,
    ) -> Tuple[float, Dict[str, float]]:
        """Validate model."""
        model.eval()
        total_loss = 0.0
        n_batches = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                features, labels, mask, _ = batch
                features = features.to(self.device).float()
                labels = labels.to(self.device).float()
                mask = mask.to(self.device).float()

                # Get the last valid label for each sequence
                batch_size = features.size(0)
                last_labels = torch.zeros(batch_size, 1, device=self.device)
                for i in range(batch_size):
                    valid_len = mask[i].sum().int().item()
                    if valid_len > 0:
                        last_labels[i] = labels[i, valid_len - 1]

                with autocast(enabled=self.config.use_amp, dtype=self.amp_dtype):
                    outputs = model(features)  # TFTLite outputs (batch, 1)
                    loss = criterion(outputs, last_labels)

                total_loss += loss.item()
                n_batches += 1

                # Collect predictions (last timestep only)
                probs = torch.sigmoid(outputs)
                all_preds.append(probs.cpu().numpy().flatten())
                all_labels.append(last_labels.cpu().numpy().flatten())

        # Flatten predictions
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # Compute metrics
        metrics = self._compute_metrics(all_preds, all_labels)

        return total_loss / n_batches, metrics

    def _evaluate(
        self,
        model: nn.Module,
        data: Any,
    ) -> Dict[str, float]:
        """Evaluate model on data."""
        criterion = nn.BCEWithLogitsLoss()
        _, metrics = self._validate(model, data, criterion)
        return metrics

    def _compute_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        from sklearn.metrics import (
            roc_auc_score,
            average_precision_score,
            f1_score,
            precision_score,
            recall_score,
        )

        binary_preds = (predictions >= threshold).astype(int)

        metrics = {}

        # AUC-ROC
        try:
            metrics["auroc"] = roc_auc_score(labels, predictions)
        except ValueError:
            metrics["auroc"] = 0.5

        # AUC-PR
        try:
            metrics["auprc"] = average_precision_score(labels, predictions)
        except ValueError:
            metrics["auprc"] = labels.mean()

        # Classification metrics
        metrics["f1"] = f1_score(labels, binary_preds, zero_division=0)
        metrics["precision"] = precision_score(labels, binary_preds, zero_division=0)
        metrics["recall"] = recall_score(labels, binary_preds, zero_division=0)
        metrics["sensitivity"] = metrics["recall"]

        # Specificity
        tn = ((binary_preds == 0) & (labels == 0)).sum()
        fp = ((binary_preds == 1) & (labels == 0)).sum()
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Approximate utility
        metrics["utility"] = self._compute_approx_utility(predictions, labels, threshold)

        return metrics

    def _compute_approx_utility(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        threshold: float,
    ) -> float:
        """Compute approximate clinical utility score."""
        binary_preds = (predictions >= threshold).astype(int)

        tp = ((binary_preds == 1) & (labels == 1)).sum()
        tn = ((binary_preds == 0) & (labels == 0)).sum()
        fp = ((binary_preds == 1) & (labels == 0)).sum()
        fn = ((binary_preds == 0) & (labels == 1)).sum()

        utility = 1.0 * tp + 0.0 * tn + -0.05 * fp + -2.0 * fn
        max_utility = tp + fn
        min_utility = -0.05 * (tn + fp) - 2.0 * (tp + fn)

        if max_utility > min_utility:
            utility_normalized = (utility - min_utility) / (max_utility - min_utility)
        else:
            utility_normalized = 0.0

        return float(utility_normalized)

    def save_final_model(self, output_path: Optional[Path] = None) -> Path:
        """
        Save the best model for deployment.

        Args:
            output_path: Path to save model

        Returns:
            Path to saved model
        """
        if output_path is None:
            output_path = Path("models/tft_lite/weights/tft_lite_v1.pt")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        best_model = self.get_best_model()
        if best_model is None:
            raise ValueError("No trained model available")

        # Save state dict
        torch.save(best_model.state_dict(), output_path)
        logger.info(f"Saved final model to {output_path}")

        # Save config
        config_path = output_path.with_suffix(".config.json")
        import json
        with open(config_path, "w") as f:
            json.dump(self.config.get_model_kwargs(), f, indent=2)
        logger.info(f"Saved config to {config_path}")

        return output_path
