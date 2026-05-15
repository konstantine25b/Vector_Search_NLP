"""Shared-backbone bi-encoder for dense retrieval (no sentence-transformers)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pool token vectors; ignore padding."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


class BiEncoder(nn.Module):
    """
    One Transformer encodes both queries and documents.
    Outputs L2-normalized embeddings suitable for cosine / dot-product InfoNCE.
    """

    def __init__(
        self,
        model_name: str,
        pooling: str = "mean",
        proj_dim: int | None = None,
    ) -> None:
        super().__init__()
        if pooling not in ("mean", "cls"):
            raise ValueError("pooling must be 'mean' or 'cls'")
        self.pooling = pooling
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = int(self.backbone.config.hidden_size)
        self._hidden = hidden
        if proj_dim is None or proj_dim == hidden:
            self.projection: nn.Module | None = None
            self.out_dim = hidden
        else:
            self.projection = nn.Linear(hidden, proj_dim, bias=True)
            self.out_dim = proj_dim

    def _pool(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return last_hidden[:, 0]
        return mean_pool(last_hidden, attention_mask)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(out.last_hidden_state, attention_mask)
        if self.projection is not None:
            pooled = self.projection(pooled)
        return F.normalize(pooled, p=2, dim=-1)

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
