import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize


class BowSearchEngine:
    def __init__(self, vectorizer: CountVectorizer, doc_matrix_norm, corpus: list[str]):
        self.vectorizer = vectorizer
        self.doc_matrix_norm = doc_matrix_norm
        self.corpus = corpus
        self._text_to_index = {t: i for i, t in enumerate(corpus)}

    @classmethod
    def fit(
        cls,
        corpus: list[str],
        max_features: int = 65_536,
        ngram_min: int = 1,
        ngram_max: int = 1,
        binary: bool = False,
    ) -> "BowSearchEngine":
        vectorizer = CountVectorizer(
            lowercase=True,
            max_features=max_features,
            ngram_range=(ngram_min, ngram_max),
            dtype=np.float32,
            binary=binary,
        )
        raw = vectorizer.fit_transform(corpus)
        doc_matrix_norm = normalize(raw, norm="l2", axis=1)
        return cls(vectorizer, doc_matrix_norm, corpus)

    def rank(self, query: str) -> np.ndarray:
        q = self.vectorizer.transform([query])
        qn = normalize(q, norm="l2", axis=1)
        sim = qn.dot(self.doc_matrix_norm.T)
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
