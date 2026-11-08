import csv

import pytest

from mdd_challen.metrics import PronunciationMetrics


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_metrics_for_correctly_detected_substitution(tmp_path) -> None:
    ground_truth = tmp_path / "ground_truth.csv"
    predictions = tmp_path / "predictions.csv"
    _write_csv(
        ground_truth,
        [{"id": "sample-1", "canonical": "a", "transcript": "b"}],
    )
    _write_csv(predictions, [{"id": "sample-1", "predict": "b"}])

    assert PronunciationMetrics.compute_f1(ground_truth, predictions) == 1.0
    assert PronunciationMetrics.compute_per(ground_truth, predictions) == 0.0
    assert PronunciationMetrics.compute_der(ground_truth, predictions) == 0.0


def test_metrics_reject_misaligned_rows(tmp_path) -> None:
    ground_truth = tmp_path / "ground_truth.csv"
    predictions = tmp_path / "predictions.csv"
    _write_csv(
        ground_truth,
        [{"id": "sample-1", "canonical": "a", "transcript": "b"}],
    )
    _write_csv(predictions, [{"id": "sample-2", "predict": "b"}])

    with pytest.raises(ValueError, match="mismatched sample ids"):
        PronunciationMetrics.compute_per(ground_truth, predictions)
