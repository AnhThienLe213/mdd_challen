import math

import torch
from torch import nn

from mdd_challen.data import PhonemeVocabulary


class SpecAugment(nn.Module):
    def __init__(
        self,
        time_mask: int = 40,
        frequency_mask: int = 64,
        time_masks: int = 2,
        frequency_masks: int = 2,
    ) -> None:
        super().__init__()
        self.time_mask = time_mask
        self.frequency_mask = frequency_mask
        self.time_masks = time_masks
        self.frequency_masks = frequency_masks

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return features
        time_steps, dimensions = features.shape
        augmented = features.clone()
        for _ in range(self.time_masks):
            width = torch.randint(min(self.time_mask, time_steps - 1) + 1, ()).item()
            start = torch.randint(max(time_steps - width, 0) + 1, ()).item()
            augmented[start : start + width, :] = 0.0
        for _ in range(self.frequency_masks):
            width = torch.randint(min(self.frequency_mask, dimensions - 1) + 1, ()).item()
            start = torch.randint(max(dimensions - width, 0) + 1, ()).item()
            augmented[:, start : start + width] = 0.0
        return augmented


class PositionalEncoding(nn.Module):
    def __init__(self, model_dimension: int, max_length: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        positions = torch.arange(max_length).unsqueeze(1).float()
        divisors = torch.exp(
            torch.arange(0, model_dimension, 2).float()
            * (-math.log(10000.0) / model_dimension)
        )
        encoding = torch.zeros(max_length, model_dimension)
        encoding[:, 0::2] = torch.sin(positions * divisors)
        encoding[:, 1::2] = torch.cos(
            positions * divisors[: encoding[:, 1::2].shape[1]]
        )
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.dropout(values + self.encoding[:, : values.size(1)])


class CTCMDDModel(nn.Module):
    """Fuses acoustic features with canonical phonemes before CTC decoding."""

    def __init__(
        self,
        vocabulary_size: int,
        audio_dimension: int,
        model_dimension: int = 256,
        attention_heads: int = 4,
        audio_layers: int = 2,
        text_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_dimension, model_dimension),
            nn.LayerNorm(model_dimension),
        )
        self.spec_augment = SpecAugment()
        self.audio_position = PositionalEncoding(model_dimension, dropout=dropout)
        self.audio_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                model_dimension,
                attention_heads,
                model_dimension * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=audio_layers,
        )
        self.text_embedding = nn.Embedding(
            vocabulary_size, model_dimension, padding_idx=PhonemeVocabulary.blank_id
        )
        self.text_position = PositionalEncoding(model_dimension, dropout=dropout)
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                model_dimension,
                attention_heads,
                model_dimension * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=text_layers,
        )
        self.fusion_attention = nn.MultiheadAttention(
            model_dimension, attention_heads, dropout=dropout, batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(model_dimension)
        self.fusion_feed_forward = nn.Sequential(
            nn.Linear(model_dimension, model_dimension * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dimension * 4, model_dimension),
        )
        self.fusion_feed_forward_norm = nn.LayerNorm(model_dimension)
        self.ctc_head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(model_dimension, vocabulary_size)
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        canonical: torch.Tensor,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, time_steps, _ = audio.shape
        padding_mask = (
            torch.arange(time_steps, device=audio.device)[None, :]
            >= audio_lengths[:, None]
        )
        augmented = torch.stack(
            [self.spec_augment(audio[index]) for index in range(batch_size)]
        )
        audio_hidden = self.audio_encoder(
            self.audio_position(self.audio_projection(augmented)),
            src_key_padding_mask=padding_mask,
        )

        canonical_padding = canonical == PhonemeVocabulary.blank_id
        text_hidden = self.text_encoder(
            self.text_position(self.text_embedding(canonical)),
            src_key_padding_mask=canonical_padding,
        )
        attended, _ = self.fusion_attention(
            audio_hidden,
            text_hidden,
            text_hidden,
            key_padding_mask=canonical_padding,
        )
        fused = self.fusion_norm(attended + audio_hidden)
        fused = self.fusion_feed_forward_norm(fused + self.fusion_feed_forward(fused))
        log_probabilities = torch.log_softmax(self.ctc_head(fused), dim=-1).permute(
            1, 0, 2
        )
        if not return_hidden:
            return log_probabilities

        valid_mask = (~padding_mask).unsqueeze(-1).to(fused.dtype)
        audio_pooled = (fused * valid_mask).sum(1) / valid_mask.sum(1).clamp(min=1)
        canonical_pooled = text_hidden.mean(dim=1)
        return log_probabilities, audio_pooled, canonical_pooled
