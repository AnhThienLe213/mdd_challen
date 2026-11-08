import csv
from pathlib import Path


class PronunciationMetrics:
    """Calculates MDD F1, phoneme error rate, and detection error rate."""

    epsilon = "<eps>"

    @classmethod
    def _align(cls, reference: list[str], hypothesis: list[str]) -> tuple[list[str], list[str]]:
        gap, match, mismatch = -1, 1, -1
        reference_length, hypothesis_length = len(reference), len(hypothesis)
        scores = [
            [0] * (reference_length + 1) for _ in range(hypothesis_length + 1)
        ]
        for index in range(hypothesis_length + 1):
            scores[index][0] = gap * index
        for index in range(reference_length + 1):
            scores[0][index] = gap * index
        for row in range(1, hypothesis_length + 1):
            for column in range(1, reference_length + 1):
                ref_token, hyp_token = reference[column - 1], hypothesis[row - 1]
                score = match if ref_token == hyp_token else (
                    gap if cls.epsilon in (ref_token, hyp_token) else mismatch
                )
                scores[row][column] = max(
                    scores[row - 1][column - 1] + score,
                    scores[row - 1][column] + gap,
                    scores[row][column - 1] + gap,
                )

        aligned_reference: list[str] = []
        aligned_hypothesis: list[str] = []
        row, column = hypothesis_length, reference_length
        while row > 0 and column > 0:
            ref_token, hyp_token = reference[column - 1], hypothesis[row - 1]
            score = match if ref_token == hyp_token else (
                gap if cls.epsilon in (ref_token, hyp_token) else mismatch
            )
            if scores[row][column] == scores[row - 1][column - 1] + score:
                aligned_reference.append(ref_token)
                aligned_hypothesis.append(hyp_token)
                row -= 1
                column -= 1
            elif scores[row][column] == scores[row][column - 1] + gap:
                aligned_reference.append(ref_token)
                aligned_hypothesis.append(cls.epsilon)
                column -= 1
            else:
                aligned_reference.append(cls.epsilon)
                aligned_hypothesis.append(hyp_token)
                row -= 1
        while column > 0:
            aligned_reference.append(reference[column - 1])
            aligned_hypothesis.append(cls.epsilon)
            column -= 1
        while row > 0:
            aligned_reference.append(cls.epsilon)
            aligned_hypothesis.append(hypothesis[row - 1])
            row -= 1
        return aligned_reference[::-1], aligned_hypothesis[::-1]

    @classmethod
    def _align_pair(
        cls, reference: str, hypothesis: str
    ) -> tuple[list[str], list[str], list[str]]:
        clean = lambda sequence: sequence.replace("*", "").replace("$", "").split()
        aligned_reference, aligned_hypothesis = cls._align(
            clean(reference), clean(hypothesis)
        )
        operations = [
            "D" if ref != cls.epsilon and hyp == cls.epsilon else
            "I" if ref == cls.epsilon and hyp != cls.epsilon else
            "S" if ref != hyp else "C"
            for ref, hyp in zip(aligned_reference, aligned_hypothesis)
        ]
        return aligned_reference, aligned_hypothesis, operations

    @staticmethod
    def _read_rows(path: str | Path) -> list[dict[str, str]]:
        with Path(path).open(newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))

    @classmethod
    def _read_pair(cls, ground_truth: str | Path, results: str | Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        truth_rows = cls._read_rows(ground_truth)
        result_rows = cls._read_rows(results)
        if len(truth_rows) != len(result_rows):
            raise ValueError(
                f"Ground truth has {len(truth_rows)} rows but predictions have "
                f"{len(result_rows)} rows."
            )
        for index, (truth, prediction) in enumerate(zip(truth_rows, result_rows)):
            if truth.get("id") != prediction.get("id"):
                raise ValueError(f"Row {index} has mismatched sample ids.")
        return truth_rows, result_rows

    @staticmethod
    def _ground_truth_sequence(row: dict[str, str], name: str) -> str:
        value = row.get(name)
        if value is None:
            value = row.get(f"{name}_phoneme")
        if value is None:
            raise ValueError(f"Ground-truth CSV is missing the {name!r} column.")
        return value

    @classmethod
    def compute_f1(cls, ground_truth: str | Path, results: str | Path) -> float:
        truth_rows, result_rows = cls._read_pair(ground_truth, results)
        correct_missed = substitution_correct = substitution_incorrect = 0
        substitution_not_detected = insertion_correct = insertion_incorrect = 0
        insertion_not_detected = deletion_correct = deletion_incorrect = 0
        deletion_not_detected = 0
        for truth, prediction in zip(truth_rows, result_rows):
            ref, human, ref_ops = cls._align_pair(
                cls._ground_truth_sequence(truth, "canonical"),
                cls._ground_truth_sequence(truth, "transcript"),
            )
            human2, predicted, human_ops = cls._align_pair(
                cls._ground_truth_sequence(truth, "transcript"),
                prediction["predict"],
            )
            ref2, _, result_ops = cls._align_pair(
                cls._ground_truth_sequence(truth, "canonical"),
                prediction["predict"],
            )
            cursor = 0
            for index, token in enumerate(ref):
                if token == cls.epsilon:
                    continue
                while cursor < len(ref2) and ref2[cursor] == cls.epsilon:
                    cursor += 1
                if cursor < len(ref2) and token == ref2[cursor]:
                    if ref_ops[index] == "D" and result_ops[cursor] == "D":
                        deletion_correct += 1
                    elif ref_ops[index] == "D" and result_ops[cursor] not in ("D", "C"):
                        deletion_incorrect += 1
                    elif ref_ops[index] == "D" and result_ops[cursor] == "C":
                        deletion_not_detected += 1
                    cursor += 1
            cursor = 0
            for index, token in enumerate(human):
                if token == cls.epsilon:
                    continue
                while cursor < len(human2) and human2[cursor] == cls.epsilon:
                    cursor += 1
                if cursor < len(human2) and token == human2[cursor]:
                    if ref_ops[index] == "C" and human_ops[cursor] != "C":
                        correct_missed += 1
                    if ref_ops[index] == "S" and human_ops[cursor] == "C":
                        substitution_correct += 1
                    elif (
                        ref_ops[index] == "S"
                        and human_ops[cursor] != "C"
                        and ref[index] != predicted[cursor]
                    ):
                        substitution_incorrect += 1
                    elif (
                        ref_ops[index] == "S"
                        and human_ops[cursor] != "C"
                        and ref[index] == predicted[cursor]
                    ):
                        substitution_not_detected += 1
                    if ref_ops[index] == "I" and human_ops[cursor] == "C":
                        insertion_correct += 1
                    elif (
                        ref_ops[index] == "I"
                        and human_ops[cursor] not in ("C", "D")
                    ):
                        insertion_incorrect += 1
                    elif ref_ops[index] == "I" and human_ops[cursor] == "D":
                        insertion_not_detected += 1
                    cursor += 1
        true_positive = (
            substitution_correct + substitution_incorrect + deletion_correct
            + deletion_incorrect + insertion_correct + insertion_incorrect
        )
        false_positive = correct_missed
        false_negative = (
            substitution_not_detected + insertion_not_detected + deletion_not_detected
        )
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    @classmethod
    def compute_per(cls, ground_truth: str | Path, results: str | Path) -> float:
        truth_rows, result_rows = cls._read_pair(ground_truth, results)
        errors = reference_length = 0
        for truth, prediction in zip(truth_rows, result_rows):
            _, _, operations = cls._align_pair(
                cls._ground_truth_sequence(truth, "transcript"),
                prediction["predict"],
            )
            errors += sum(operation in ("S", "D", "I") for operation in operations)
            reference_length += sum(operation in ("S", "D", "C") for operation in operations)
        return errors / reference_length if reference_length else 0.0

    @classmethod
    def compute_der(cls, ground_truth: str | Path, results: str | Path) -> float:
        truth_rows, result_rows = cls._read_pair(ground_truth, results)
        sub_correct = sub_incorrect = sub_not_detected = 0
        ins_correct = ins_incorrect = ins_not_detected = 0
        del_correct = del_incorrect = del_not_detected = 0
        for truth, prediction in zip(truth_rows, result_rows):
            ref, human, ref_ops = cls._align_pair(
                cls._ground_truth_sequence(truth, "canonical"),
                cls._ground_truth_sequence(truth, "transcript"),
            )
            human2, predicted, human_ops = cls._align_pair(
                cls._ground_truth_sequence(truth, "transcript"),
                prediction["predict"],
            )
            ref2, _, result_ops = cls._align_pair(
                cls._ground_truth_sequence(truth, "canonical"),
                prediction["predict"],
            )
            cursor = 0
            for index, token in enumerate(ref):
                if token == cls.epsilon:
                    continue
                while cursor < len(ref2) and ref2[cursor] == cls.epsilon:
                    cursor += 1
                if cursor < len(ref2) and token == ref2[cursor]:
                    if ref_ops[index] == "D" and result_ops[cursor] == "D":
                        del_correct += 1
                    elif ref_ops[index] == "D" and result_ops[cursor] not in ("D", "C"):
                        del_incorrect += 1
                    elif ref_ops[index] == "D" and result_ops[cursor] == "C":
                        del_not_detected += 1
                    cursor += 1
            cursor = 0
            for index, token in enumerate(human):
                if token == cls.epsilon:
                    continue
                while cursor < len(human2) and human2[cursor] == cls.epsilon:
                    cursor += 1
                if cursor < len(human2) and token == human2[cursor]:
                    if ref_ops[index] == "S" and human_ops[cursor] == "C":
                        sub_correct += 1
                    elif ref_ops[index] == "S" and human_ops[cursor] != "C" and ref[index] != predicted[cursor]:
                        sub_incorrect += 1
                    elif ref_ops[index] == "S" and human_ops[cursor] != "C" and ref[index] == predicted[cursor]:
                        sub_not_detected += 1
                    if ref_ops[index] == "I" and human_ops[cursor] == "C":
                        ins_correct += 1
                    elif ref_ops[index] == "I" and human_ops[cursor] not in ("C", "D"):
                        ins_incorrect += 1
                    elif ref_ops[index] == "I" and human_ops[cursor] == "D":
                        ins_not_detected += 1
                    cursor += 1
        true_positive = (
            sub_correct + sub_incorrect + del_correct + del_incorrect
            + ins_correct + ins_incorrect
        )
        false_negative = sub_not_detected + ins_not_detected + del_not_detected
        detected_errors = sub_incorrect + del_incorrect + ins_incorrect
        return detected_errors / (true_positive + false_negative) if true_positive + false_negative else 0.0
