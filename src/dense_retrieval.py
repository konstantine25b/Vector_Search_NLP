"""Dense (embedding) retrieval and metrics using the same interface as TF-IDF evaluation."""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoTokenizer

from tqdm import tqdm

from src.biencoder import BiEncoder
from src.tfidf_retrieval import evaluate_ranking

# Re-export for scripts that import corpus helpers from one place.
__all__ = ["DenseSearchEngine", "evaluate_ranking", "encode_corpus"]


class DenseSearchEngine:
    """
    Precomputed document embeddings + on-the-fly query encoding.
    `rank` returns dot products (= cosine similarities) versus all indexed passages.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        encoder: BiEncoder,
        corpus: list[str],
        doc_embeddings: np.ndarray,
        device: torch.device,
        max_query_length: int = 64,
    ) -> None:
        if doc_embeddings.ndim != 2:
            raise ValueError("doc_embeddings must be [num_docs, dim]")
        if doc_embeddings.shape[0] != len(corpus):
            raise ValueError("corpus length must match doc_embeddings rows")
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.corpus = corpus
        self.doc_embeddings = np.asarray(doc_embeddings, dtype=np.float32)
        self.device = device
        self.max_query_length = max_query_length
        self._text_to_index = {t: i for i, t in enumerate(corpus)}

    def rank(self, query: str) -> np.ndarray:
        self.encoder.eval()
        toks = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=self.max_query_length,
            return_tensors="pt",
        )
        input_ids = toks["input_ids"].to(self.device)
        attn = toks["attention_mask"].to(self.device)
        with torch.no_grad():
            q = self.encoder.encode(input_ids, attn).detach().cpu().numpy().astype(np.float32)
        scores = self.doc_embeddings @ q.T
        return scores.ravel()

    def gold_rank(self, query: str, gold_docs: set[str]) -> int | None:
        scores = self.rank(query)
        indices = [self._text_to_index[d] for d in gold_docs if d in self._text_to_index]
        if not indices:
            return None
        n = scores.shape[0]
        idx_all = np.arange(n)
        best: int | None = None
        for g in indices:
            s = scores[g]
            r = int(np.sum(scores > s) + np.sum((scores == s) & (idx_all < g)))
            if best is None or r < best:
                best = r
        return best


def encode_corpus(
    encoder: BiEncoder,
    tokenizer: AutoTokenizer,
    texts: list[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
    *,
    show_progress: bool = True,
) -> np.ndarray:
    """Encode a fixed corpus; returns float32 numpy [N, dim] (L2-normalized rows)."""
    encoder.eval()
    out_list: list[np.ndarray] = []
    n = len(texts)
    n_batch = (n + batch_size - 1) // batch_size if n else 0
    batch_starts = range(0, n, batch_size)
    if show_progress and n_batch > 0:
        batch_starts = tqdm(
            batch_starts,
            desc="Encode corpus",
            total=n_batch,
            unit="batch",
        )
    with torch.no_grad():
        for start in batch_starts:
            chunk = texts[start : start + batch_size]
            batch = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            emb = encoder.encode(ids, mask).detach().cpu().numpy().astype(np.float32)
            out_list.append(emb)
    if not out_list:
        hidden = encoder.out_dim
        return np.zeros((0, hidden), dtype=np.float32)
    return np.concatenate(out_list, axis=0)
