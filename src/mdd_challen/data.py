import json
import os
import random
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pandas as pd
from sklearn.model_selection import train_test_split

from mdd_challen.config import DataConfig, ModelConfig


class PhonemeVocabulary:
    """Owns phoneme/id conversion and the CTC blank-token convention."""

    blank_token = "<BLANK>"
    blank_id = 0

    def __init__(self, phonemes: list[str]) -> None:
        self.tokens = [self.blank_token, *sorted(set(phonemes) - {self.blank_token})]
        self.phoneme_to_id = {token: index for index, token in enumerate(self.tokens)}
        self.id_to_phoneme = dict(enumerate(self.tokens))

    @classmethod
    def from_frames(cls, *frames: pd.DataFrame) -> "PhonemeVocabulary":
        phonemes: set[str] = set()
        for frame in frames:
            for column in ("canonical_phoneme", "transcript_phoneme"):
                if column not in frame:
                    continue
                for sequence in frame[column].fillna("").astype(str):
                    phonemes.update(sequence.split())
        if not phonemes:
            raise ValueError("No phonemes were found in the supplied datasets.")
        return cls(list(phonemes))

    def encode(self, text: Any) -> list[int]:
        tokens = str(text).split()
        unknown = sorted(set(tokens) - self.phoneme_to_id.keys())
        if unknown:
            raise ValueError(f"Unknown phoneme(s): {', '.join(unknown)}")
        return [self.phoneme_to_id[token] for token in tokens]

    def decode_ctc(self, ids: list[int]) -> list[str]:
        decoded: list[str] = []
        previous: int | None = None
        for token_id in ids:
            if token_id != previous and token_id != self.blank_id:
                token = self.id_to_phoneme.get(token_id)
                if token is not None:
                    decoded.append(token)
            previous = token_id
        return decoded

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.phoneme_to_id, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "PhonemeVocabulary":
        mapping = json.loads(path.read_text(encoding="utf-8"))
        ordered = sorted(mapping, key=mapping.__getitem__)
        if not ordered or ordered[0] != cls.blank_token:
            raise ValueError(f"{path} does not contain a valid CTC vocabulary.")
        return cls(ordered[1:])


class AudioFeatureExtractor:
    """Extracts fixed Wav2Vec2 representations; the pretrained model stays frozen."""

    def __init__(self, config: ModelConfig, device: torch.device) -> None:
        os.environ.setdefault("USE_TF", "0")
        from transformers import Wav2Vec2Model, Wav2Vec2Processor

        self.config = config
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(config.pretrained_model)
        self.model = Wav2Vec2Model.from_pretrained(config.pretrained_model).to(device)
        self.model.requires_grad_(False)
        self.model.eval()
        self.hidden_size = int(self.model.config.hidden_size)

    @torch.no_grad()
    def extract(self, wav_path: str | Path, speed: float = 1.0) -> torch.Tensor:
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(wav_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.config.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.config.sample_rate
            )
        if speed != 1.0:
            waveform = torchaudio.functional.resample(
                waveform,
                self.config.sample_rate,
                int(self.config.sample_rate * speed),
            )
        inputs = self.processor(
            waveform.squeeze().numpy(),
            sampling_rate=self.config.sample_rate,
            return_tensors="pt",
        )
        features = self.model(inputs.input_values.to(self.device)).last_hidden_state
        return features.squeeze(0).cpu()


class PronunciationDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        vocabulary: PhonemeVocabulary,
        extractor: AudioFeatureExtractor,
        training: bool,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.vocabulary = vocabulary
        self.extractor = extractor
        self.training = training

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        row = self.frame.iloc[index]
        speed = (
            random.choice((0.9, 1.0, 1.1))
            if self.training and random.random() < 0.5
            else 1.0
        )
        features = self.extractor.extract(row["wav_path"], speed=speed)
        canonical = torch.tensor(
            self.vocabulary.encode(row["canonical_phoneme"]), dtype=torch.long
        )
        if not self.training:
            return row["id"], features, canonical
        transcript = torch.tensor(
            self.vocabulary.encode(row["transcript_phoneme"]), dtype=torch.long
        )
        mispronounced = torch.tensor(
            row["is_mispron"], dtype=torch.float32
        )
        return features, canonical, transcript, mispronounced


def training_collate(batch: list[tuple[torch.Tensor, ...]]) -> tuple[torch.Tensor, ...]:
    features, canonical, transcripts, mispronounced = zip(*batch)
    feature_lengths = torch.tensor([item.size(0) for item in features], dtype=torch.long)
    target_lengths = torch.tensor([item.size(0) for item in transcripts], dtype=torch.long)
    return (
        pad_sequence(features, batch_first=True, padding_value=0.0),
        feature_lengths,
        pad_sequence(canonical, batch_first=True, padding_value=PhonemeVocabulary.blank_id),
        pad_sequence(transcripts, batch_first=True, padding_value=PhonemeVocabulary.blank_id),
        target_lengths,
        torch.stack(mispronounced),
    )


def inference_collate(
    batch: list[tuple[Any, torch.Tensor, torch.Tensor]],
) -> tuple[list[Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    identifiers, features, canonical = zip(*batch)
    feature_lengths = torch.tensor([item.size(0) for item in features], dtype=torch.long)
    return (
        list(identifiers),
        pad_sequence(features, batch_first=True, padding_value=0.0),
        pad_sequence(canonical, batch_first=True, padding_value=PhonemeVocabulary.blank_id),
        feature_lengths,
    )


class MDDDataModule:
    """Loads source CSVs and provides train, validation, and public-test loaders."""

    def __init__(
        self,
        config: DataConfig,
        extractor: AudioFeatureExtractor,
    ) -> None:
        self.config = config
        self.extractor = extractor
        self.vocabulary: PhonemeVocabulary
        self.train_loader: DataLoader
        self.validation_loader: DataLoader
        self.test_loader: DataLoader

    @staticmethod
    def _read_split(data_dir: Path, audio_dir: Path, prefix: str) -> pd.DataFrame:
        text_path = data_dir / f"{prefix}.csv"
        phone_path = data_dir / f"{prefix}_phones.csv"
        text_frame = pd.read_csv(text_path)
        phone_frame = pd.read_csv(phone_path)
        frame = text_frame.merge(phone_frame, on="id", suffixes=("_text", "_phoneme"))
        if "canonical_phoneme" not in frame and "canonical" in frame:
            frame["canonical_phoneme"] = frame["canonical"]
        if "transcript_phoneme" not in frame and "transcript" in frame:
            frame["transcript_phoneme"] = frame["transcript"]
        if "path_text" not in frame and "path" in frame:
            frame["path_text"] = frame["path"]
        required = {"id", "path_text", "canonical_phoneme", "transcript_phoneme"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{prefix} data is missing columns: {', '.join(sorted(missing))}")
        frame["wav_path"] = frame["path_text"].map(
            lambda value: str(audio_dir / (Path(str(value)).name.removesuffix(".wav") + ".wav"))
        )
        absent = [path for path in frame["wav_path"] if not Path(path).is_file()]
        if absent:
            raise FileNotFoundError(f"Missing audio file: {absent[0]} ({len(absent)} missing total)")
        frame["is_mispron"] = (
            frame["canonical_phoneme"].fillna("").astype(str).str.strip()
            != frame["transcript_phoneme"].fillna("").astype(str).str.strip()
        ).astype("int64")
        return frame

    def setup(self) -> None:
        train_frame = self._read_split(
            self.config.data_dir, self.config.audio_dir, "train"
        )
        test_frame = self._read_split(
            self.config.data_dir, self.config.test_audio_dir, "public_test"
        )
        self.vocabulary = PhonemeVocabulary.from_frames(train_frame, test_frame)
        train_frame, validation_frame = train_test_split(
            train_frame,
            test_size=self.config.validation_fraction,
            random_state=self.config.random_seed,
            stratify=train_frame["is_mispron"],
        )
        weights = train_frame["is_mispron"].map(
            lambda value: self.config.mispronunciation_weight if value else 1.0
        ).to_numpy()
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )
        common = {
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
        }
        self.train_loader = DataLoader(
            PronunciationDataset(train_frame, self.vocabulary, self.extractor, True),
            sampler=sampler,
            collate_fn=training_collate,
            **common,
        )
        self.validation_loader = DataLoader(
            PronunciationDataset(validation_frame, self.vocabulary, self.extractor, True),
            shuffle=False,
            collate_fn=training_collate,
            **common,
        )
        self.test_loader = DataLoader(
            PronunciationDataset(test_frame, self.vocabulary, self.extractor, False),
            batch_size=1,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=inference_collate,
        )
