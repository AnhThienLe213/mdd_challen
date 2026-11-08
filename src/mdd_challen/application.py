import random
from pathlib import Path

import torch
import numpy as np

from mdd_challen.config import DataConfig, ModelConfig, TrainingConfig
from mdd_challen.data import AudioFeatureExtractor, MDDDataModule
from mdd_challen.metrics import PronunciationMetrics
from mdd_challen.model import CTCMDDModel
from mdd_challen.training import GreedyCTCDecoder, MDDTrainer


class MDDApplication:
    """Coordinates model construction, data setup, training, and evaluation."""

    def __init__(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        checkpoint: Path | None = None,
    ) -> None:
        self.data_config = data_config
        self.model_config = model_config
        self.training_config = training_config
        self.checkpoint = checkpoint

    def run(self) -> dict[str, float] | None:
        self._seed_everything(self.data_config.random_seed)
        device = self._resolve_device(self.model_config.device)
        print(f"Device: {device}")

        extractor = AudioFeatureExtractor(self.model_config, device)
        data = MDDDataModule(self.data_config, extractor)
        data.setup()
        data.vocabulary.save(self.data_config.checkpoint_dir / "phoneme_vocab.json")
        print(
            f"Train batches: {len(data.train_loader)} | "
            f"Validation batches: {len(data.validation_loader)} | "
            f"Vocabulary size: {len(data.vocabulary.tokens)}"
        )

        model = CTCMDDModel(
            vocabulary_size=len(data.vocabulary.tokens),
            audio_dimension=extractor.hidden_size,
            model_dimension=self.model_config.model_dimension,
            attention_heads=self.model_config.attention_heads,
            audio_layers=self.model_config.audio_layers,
            text_layers=self.model_config.text_layers,
            dropout=self.model_config.dropout,
        ).to(device)
        print(f"Model parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")

        trainer = MDDTrainer(
            model,
            data.vocabulary,
            self.training_config,
            device,
            self.data_config.checkpoint_dir,
            GreedyCTCDecoder(data.vocabulary),
        )
        if self.checkpoint is None:
            trainer.fit(data.train_loader, data.validation_loader)
            checkpoint_path = self.data_config.checkpoint_dir / "best_model.pt"
        else:
            checkpoint_path = self.checkpoint
        checkpoint = trainer.load_checkpoint(checkpoint_path)

        predictions = trainer.predict(data.test_loader)
        self.data_config.output_dir.mkdir(parents=True, exist_ok=True)
        result_path = self.data_config.output_dir / "results.csv"
        predictions.to_csv(result_path, index=False)
        print(f"Saved {len(predictions)} predictions to {result_path}")

        ground_truth = self.data_config.data_dir / "public_test_phones.csv"
        metrics = None
        if ground_truth.is_file():
            metrics = {
                "f1": PronunciationMetrics.compute_f1(ground_truth, result_path),
                "per": PronunciationMetrics.compute_per(ground_truth, result_path),
                "der": PronunciationMetrics.compute_der(ground_truth, result_path),
            }
            metrics["score"] = (
                0.5 * metrics["f1"]
                + 0.4 * (1 - metrics["der"])
                + 0.1 * (1 - metrics["per"])
            )
            print(
                f"F1: {metrics['f1']:.4f} | DER: {metrics['der']:.4f} | "
                f"PER: {metrics['per']:.4f} | Score: {metrics['score']:.4f}"
            )
        print(
            f"Loaded checkpoint from epoch {checkpoint['epoch']} "
            f"(validation loss {checkpoint['val_loss']:.4f})"
        )
        return metrics

    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        if device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return device

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
