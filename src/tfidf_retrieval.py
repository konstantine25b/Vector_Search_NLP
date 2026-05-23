import json
from collections import defaultdict
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_unique_corpus(paths: list[Path]) -> list[str]:
    seen: set[str] = set()
    corpus: list[str] = []
    for path in paths:
        for row in load_jsonl(path):
            text = row["document"]
            if text not in seen:
                seen.add(text)
                corpus.append(text)
    return corpus


class RankingEngine(Protocol):
    def gold_rank(self, query: str, gold_docs: set[str]) -> int | None: ...


def group_queries_by_id(rows: list[dict]) -> tuple[dict, dict]:
    gold: dict = defaultdict(set)
    qtext: dict = {}
    for row in rows:
        qid = row["query_id"]
        gold[qid].add(row["document"])
        qtext[qid] = row["query"]
    return dict(gold), qtext


class TfidfSearchEngine:
    def __init__(self, vectorizer: TfidfVectorizer, doc_matrix, corpus: list[str]):
        self.vectorizer = vectorizer
        self.doc_matrix = doc_matrix
        self.corpus = corpus
        self._text_to_index = {t: i for i, t in enumerate(corpus)}

    @classmethod
    def fit(
        cls,
        corpus: list[str],
        max_features: int = 65_536,
        ngram_min: int = 1,
        ngram_max: int = 1,
    ) -> "TfidfSearchEngine":
        vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=max_features,
            ngram_range=(ngram_min, ngram_max),
            dtype=np.float32,
            sublinear_tf=True,
        )
        doc_matrix = vectorizer.fit_transform(corpus)
        return cls(vectorizer, doc_matrix, corpus)

    def rank(self, query: str) -> np.ndarray:
        q = self.vectorizer.transform([query])
        sim = q.dot(self.doc_matrix.T)
        return np.asarray(sim.toarray()).ravel()

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


def recall_at_k(rank_positions: list[int | None], k: int) -> float:
    hits = sum(1 for r in rank_positions if r is not None and r < k)
    return hits / max(len(rank_positions), 1)


def mean_reciprocal_rank(rank_positions: list[int | None]) -> float:
    scores = []
    for r in rank_positions:
        if r is None:
            scores.append(0.0)
        else:
            scores.append(1.0 / (r + 1))
    return float(np.mean(scores)) if scores else 0.0


def evaluate_ranking(
    engine: RankingEngine,
    gold_by_qid: dict,
    query_text: dict,
    ks: list[int],
    query_ids: list | None = None,
) -> dict:
    rank_first_hit: list[int | None] = []
    qids = query_ids if query_ids is not None else sorted(gold_by_qid.keys())
    for qid in qids:
        gold_docs = gold_by_qid[qid]
        q = query_text[qid]
        best_rank = engine.gold_rank(q, gold_docs)
        rank_first_hit.append(best_rank)
    out: dict = {"n_queries": len(rank_first_hit), "mrr": mean_reciprocal_rank(rank_first_hit)}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(rank_first_hit, k)
    return out
