import torch
import pandas as pd
import pytest

from mdd_challen.data import (
    MDDDataModule,
    PhonemeVocabulary,
    inference_collate,
    training_collate,
)


def test_vocabulary_round_trip_and_ctc_collapse() -> None:
    frame = pd.DataFrame(
        {"canonical_phoneme": ["b a"], "transcript_phoneme": ["b b a"]}
    )
    vocabulary = PhonemeVocabulary.from_frames(frame)

    encoded = vocabulary.encode("b b a")

    assert vocabulary.decode_ctc([encoded[0], encoded[0], 0, encoded[0], encoded[2]]) == [
        "b",
        "b",
        "a",
    ]
    with pytest.raises(ValueError, match="Unknown phoneme"):
        vocabulary.encode("unknown")


def test_collators_pad_sequences_and_preserve_lengths() -> None:
    training_batch = [
        (
            torch.ones(3, 2),
            torch.tensor([1]),
            torch.tensor([1, 2]),
            torch.tensor(0.0),
        ),
        (
            torch.ones(2, 2),
            torch.tensor([2, 1]),
            torch.tensor([2]),
            torch.tensor(1.0),
        ),
    ]

    audio, audio_lengths, canonical, targets, target_lengths, labels = training_collate(
        training_batch
    )

    assert audio.shape == (2, 3, 2)
    assert audio_lengths.tolist() == [3, 2]
    assert canonical.tolist() == [[1, 0], [2, 1]]
    assert targets.tolist() == [[1, 2], [2, 0]]
    assert target_lengths.tolist() == [2, 1]
    assert labels.tolist() == [0.0, 1.0]

    ids, features, padded_canonical, lengths = inference_collate(
        [("a", torch.ones(2, 3), torch.tensor([1])), ("b", torch.ones(1, 3), torch.tensor([2, 1]))]
    )
    assert ids == ["a", "b"]
    assert features.shape == (2, 2, 3)
    assert padded_canonical.tolist() == [[1, 0], [2, 1]]
    assert lengths.tolist() == [2, 1]


def test_read_split_merges_phone_labels_and_resolves_audio_path(tmp_path) -> None:
    data_dir = tmp_path / "tables"
    audio_dir = tmp_path / "audio"
    data_dir.mkdir()
    audio_dir.mkdir()
    (audio_dir / "recording.wav").touch()
    pd.DataFrame(
        [{"id": "sample-1", "path": "recording.wav", "canonical": "a", "transcript": "b"}]
    ).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame(
        [{"id": "sample-1", "canonical": "a", "transcript": "b"}]
    ).to_csv(data_dir / "train_phones.csv", index=False)

    frame = MDDDataModule._read_split(data_dir, audio_dir, "train")

    assert frame.loc[0, "canonical_phoneme"] == "a"
    assert frame.loc[0, "transcript_phoneme"] == "b"
    assert frame.loc[0, "wav_path"] == str(audio_dir / "recording.wav")
    assert frame.loc[0, "is_mispron"] == 1
