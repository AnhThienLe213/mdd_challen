import torch
import torch.nn.functional as functional
from torch import nn

from mdd_challen.data import PhonemeVocabulary


class CompositeMDDLoss(nn.Module):
    """Combines transcript CTC, canonical-copy penalty, and contrastive loss."""

    def __init__(
        self,
        canonical_penalty_weight: float = 0.1,
        contrastive_weight: float = 0.05,
        contrastive_margin: float = 1.0,
    ) -> None:
        super().__init__()
        self.ctc = nn.CTCLoss(
            blank=PhonemeVocabulary.blank_id,
            reduction="mean",
            zero_infinity=True,
        )
        self.canonical_penalty_weight = canonical_penalty_weight
        self.contrastive_weight = contrastive_weight
        self.contrastive_margin = contrastive_margin

    def forward(
        self,
        log_probabilities: torch.Tensor,
        targets: torch.Tensor,
        feature_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        canonical: torch.Tensor,
        audio_pooled: torch.Tensor,
        canonical_pooled: torch.Tensor,
        is_mispronounced: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        ctc_loss = self.ctc(
            log_probabilities, targets, feature_lengths, target_lengths
        )
        canonical_lengths = (canonical != PhonemeVocabulary.blank_id).sum(dim=1)
        canonical_lengths = canonical_lengths.clamp(min=1)
        canonical_loss = self.ctc(
            log_probabilities, canonical, feature_lengths, canonical_lengths
        )
        penalty = -self.canonical_penalty_weight * canonical_loss.clamp(max=5.0)

        distance = functional.pairwise_distance(audio_pooled, canonical_pooled)
        correct_mask = (is_mispronounced == 0).to(distance.dtype)
        mispronounced_mask = is_mispronounced.to(distance.dtype)
        contrastive = (
            correct_mask * distance.pow(2)
            + mispronounced_mask
            * functional.relu(self.contrastive_margin - distance).pow(2)
        ).mean()
        contrastive = self.contrastive_weight * contrastive
        total = ctc_loss + penalty + contrastive
        breakdown = {
            "ctc": ctc_loss.item(),
            "penalty": penalty.item(),
            "contrastive": contrastive.item(),
        }
        return total, breakdown
