"""BiLSTM bi-encoder for dense retrieval — trained from scratch, no pretrained model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMEncoder(nn.Module):
    """
    Embedding → BiLSTM → mean pool → projection → L2 normalize.

    Same interface as BiEncoder (encode / forward) so the training loop,
    evaluation, and demo scripts can use it as a drop-in replacement.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
        proj_dim: int = 256,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_dim * 2  # bidirectional doubles the output
        self.projection = nn.Linear(lstm_out_dim, proj_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = proj_dim

    def _mean_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.dropout(self.embedding(input_ids))

        lengths = attention_mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False,
        )
        lstm_out, _ = self.lstm(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        pooled = self._mean_pool(unpacked, attention_mask)
        projected = self.projection(pooled)
        return F.normalize(projected, p=2, dim=-1)

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.encode(query_input_ids, query_attention_mask)
        d = self.encode(doc_input_ids, doc_attention_mask)
        return q, d
