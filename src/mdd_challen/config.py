from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    data_dir: Path
    audio_dir: Path
    test_audio_dir: Path
    output_dir: Path = Path("outputs")
    checkpoint_dir: Path = Path("checkpoints")
    validation_fraction: float = 0.1
    batch_size: int = 8
    mispronunciation_weight: float = 3.0
    random_seed: int = 42
    num_workers: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self.mispronunciation_weight <= 0:
            raise ValueError("mispronunciation_weight must be positive.")


@dataclass(frozen=True)
class ModelConfig:
    pretrained_model: str = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
    model_dimension: int = 256
    attention_heads: int = 4
    audio_layers: int = 2
    text_layers: int = 1
    dropout: float = 0.2
    sample_rate: int = 16_000
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.model_dimension % self.attention_heads:
            raise ValueError("model_dimension must be divisible by attention_heads.")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 50
    model_learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    minimum_learning_rate: float = 1e-6
    patience: int = 8
    gradient_clip_norm: float = 1.0
    canonical_penalty_weight: float = 0.1
    contrastive_weight: float = 0.05
    unfreeze_schedule: dict[int, int | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive.")
        if self.patience < 1:
            raise ValueError("patience must be positive.")
        if self.model_learning_rate <= 0:
            raise ValueError("model_learning_rate must be positive.")
        if self.unfreeze_schedule:
            raise ValueError(
                "Wav2Vec2 is used as a frozen feature extractor; "
                "unfreeze_schedule is not supported."
            )
