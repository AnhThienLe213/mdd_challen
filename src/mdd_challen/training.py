from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from mdd_challen.config import TrainingConfig
from mdd_challen.data import PhonemeVocabulary
from mdd_challen.losses import CompositeMDDLoss
from mdd_challen.model import CTCMDDModel


class DecoderStrategy(Protocol):
    def decode(self, log_probabilities: torch.Tensor) -> list[str]: ...


class GreedyCTCDecoder:
    def __init__(self, vocabulary: PhonemeVocabulary) -> None:
        self.vocabulary = vocabulary

    def decode(self, log_probabilities: torch.Tensor) -> list[str]:
        predictions: list[str] = []
        for batch_index in range(log_probabilities.size(1)):
            token_ids = log_probabilities[:, batch_index, :].argmax(dim=-1).cpu().tolist()
            predictions.append(" ".join(self.vocabulary.decode_ctc(token_ids)))
        return predictions


class MDDTrainer:
    """Owns the training loop, early stopping, checkpointing, and prediction."""

    def __init__(
        self,
        model: CTCMDDModel,
        vocabulary: PhonemeVocabulary,
        config: TrainingConfig,
        device: torch.device,
        checkpoint_dir: Path,
        decoder: DecoderStrategy,
    ) -> None:
        self.model = model
        self.vocabulary = vocabulary
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.decoder = decoder
        self.criterion = CompositeMDDLoss(
            canonical_penalty_weight=config.canonical_penalty_weight,
            contrastive_weight=config.contrastive_weight,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.model_learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=config.minimum_learning_rate,
        )

    def _run_epoch(self, loader: DataLoader, training: bool) -> dict[str, float]:
        if len(loader) == 0:
            raise ValueError("Cannot run an epoch with an empty data loader.")
        self.model.train(training)
        totals = {"loss": 0.0, "ctc": 0.0, "penalty": 0.0, "contrastive": 0.0}
        context = torch.enable_grad if training else torch.no_grad
        with context():
            for audio, feature_lengths, canonical, targets, target_lengths, labels in loader:
                audio = audio.to(self.device)
                feature_lengths = feature_lengths.to(self.device)
                canonical = canonical.to(self.device)
                targets = targets.to(self.device)
                target_lengths = target_lengths.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(
                    audio, feature_lengths, canonical, return_hidden=True
                )
                if not isinstance(outputs, tuple):
                    raise RuntimeError("The model did not return training representations.")
                log_probabilities, audio_pooled, canonical_pooled = outputs
                loss, breakdown = self.criterion(
                    log_probabilities,
                    targets,
                    feature_lengths,
                    target_lengths,
                    canonical,
                    audio_pooled,
                    canonical_pooled,
                    labels,
                )
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_norm
                    )
                    self.optimizer.step()
                totals["loss"] += loss.item()
                for name, value in breakdown.items():
                    totals[name] += value
        return {name: value / len(loader) for name, value in totals.items()}

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
    ) -> list[dict[str, Any]]:
        best_validation_loss = float("inf")
        patience_counter = 0
        history: list[dict[str, Any]] = []
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._run_epoch(train_loader, training=True)
            validation_metrics = self._run_epoch(validation_loader, training=False)
            self.scheduler.step()
            learning_rate = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:02d}/{self.config.epochs} | "
                f"Train {train_metrics['loss']:.4f} "
                f"(ctc={train_metrics['ctc']:.3f} "
                f"pen={train_metrics['penalty']:.3f} "
                f"con={train_metrics['contrastive']:.3f}) | "
                f"Val {validation_metrics['loss']:.4f} | LR {learning_rate:.2e}"
            )
            history.append(
                {
                    "epoch": epoch,
                    **{f"train_{key}": value for key, value in train_metrics.items()},
                    **{f"val_{key}": value for key, value in validation_metrics.items()},
                }
            )

            if validation_metrics["loss"] < best_validation_loss:
                best_validation_loss = validation_metrics["loss"]
                patience_counter = 0
                torch.save(
                    {
                        "model_state": self.model.state_dict(),
                        "phoneme_to_id": self.vocabulary.phoneme_to_id,
                        "epoch": epoch,
                        "val_loss": best_validation_loss,
                    },
                    self.checkpoint_dir / "best_model.pt",
                )
                print(f"  Saved best model (val={best_validation_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

        pd.DataFrame(history).to_csv(
            self.checkpoint_dir / "train_history.csv", index=False
        )
        return history

    def load_checkpoint(self, checkpoint_path: Path) -> dict[str, Any]:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if checkpoint["phoneme_to_id"] != self.vocabulary.phoneme_to_id:
            raise ValueError("Checkpoint vocabulary does not match the current dataset.")
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        return checkpoint

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> pd.DataFrame:
        self.model.eval()
        rows: list[dict[str, Any]] = []
        for identifiers, audio, canonical, feature_lengths in loader:
            log_probabilities = self.model(
                audio.to(self.device),
                feature_lengths.to(self.device),
                canonical.to(self.device),
            )
            if not isinstance(log_probabilities, torch.Tensor):
                raise RuntimeError("The model returned unexpected inference outputs.")
            predictions = self.decoder.decode(log_probabilities)
            rows.extend(
                {"id": identifier, "predict": prediction}
                for identifier, prediction in zip(identifiers, predictions)
            )
        return pd.DataFrame(rows, columns=["id", "predict"])
