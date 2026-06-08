"""BM25 retrieval engine — same rank() / gold_rank() interface as TF-IDF and BoW."""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    """Lowercase + split on non-alphanumeric (mirrors the TF-IDF analyzer default)."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25SearchEngine:
    """
    BM25Okapi over a fixed corpus.

    Same interface as TfidfSearchEngine and BowSearchEngine so evaluate_ranking
    and all scripts work as drop-in replacements.
    """

    def __init__(self, bm25: BM25Okapi, corpus: list[str]) -> None:
        self.bm25 = bm25
        self.corpus = corpus
        self._text_to_index = {t: i for i, t in enumerate(corpus)}

    @classmethod
    def fit(cls, corpus: list[str]) -> "BM25SearchEngine":
        tokenized = [_tokenize(doc) for doc in corpus]
        bm25 = BM25Okapi(tokenized)
        return cls(bm25, corpus)

    def rank(self, query: str) -> np.ndarray:
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        return np.asarray(scores, dtype=np.float32)

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
