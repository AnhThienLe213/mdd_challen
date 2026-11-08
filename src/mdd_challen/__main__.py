import argparse
from pathlib import Path

from mdd_challen.application import MDDApplication
from mdd_challen.config import DataConfig, ModelConfig, TrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train or run the CTC mispronunciation-detection pipeline."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--test-audio-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--pretrained-model",
        default="nguyenvulebinh/wav2vec2-base-vietnamese-250h",
    )
    args = parser.parse_args()

    application = MDDApplication(
        data_config=DataConfig(
            data_dir=args.data_dir,
            audio_dir=args.audio_dir,
            test_audio_dir=args.test_audio_dir,
            output_dir=args.output_dir,
            checkpoint_dir=args.checkpoint_dir,
            batch_size=args.batch_size,
        ),
        model_config=ModelConfig(
            pretrained_model=args.pretrained_model,
            device=args.device,
        ),
        training_config=TrainingConfig(epochs=args.epochs),
        checkpoint=args.checkpoint,
    )
    application.run()


if __name__ == "__main__":
    main()
