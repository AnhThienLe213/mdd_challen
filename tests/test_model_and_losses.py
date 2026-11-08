import torch

from mdd_challen.losses import CompositeMDDLoss
from mdd_challen.model import CTCMDDModel


def test_model_training_outputs_and_composite_loss_backpropagate() -> None:
    model = CTCMDDModel(
        vocabulary_size=5,
        audio_dimension=4,
        model_dimension=8,
        attention_heads=2,
        audio_layers=1,
        text_layers=1,
        dropout=0.0,
    )
    audio = torch.randn(2, 6, 4)
    audio_lengths = torch.tensor([6, 5])
    canonical = torch.tensor([[1, 2], [2, 0]])
    targets = torch.tensor([[1, 2], [2, 0]])
    target_lengths = torch.tensor([2, 1])
    labels = torch.tensor([0.0, 1.0])

    outputs = model(audio, audio_lengths, canonical, return_hidden=True)
    assert isinstance(outputs, tuple)
    log_probabilities, audio_pooled, canonical_pooled = outputs
    assert log_probabilities.shape == (6, 2, 5)
    assert audio_pooled.shape == canonical_pooled.shape == (2, 8)

    loss, parts = CompositeMDDLoss()(
        log_probabilities,
        targets,
        audio_lengths,
        target_lengths,
        canonical,
        audio_pooled,
        canonical_pooled,
        labels,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert set(parts) == {"ctc", "penalty", "contrastive"}
    assert model.ctc_head[-1].weight.grad is not None
