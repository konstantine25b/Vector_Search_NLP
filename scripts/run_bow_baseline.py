import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from sklearn.preprocessing import normalize

from src.bow_retrieval import BowSearchEngine
from src.tfidf_retrieval import (
    build_unique_corpus,
    evaluate_ranking,
    group_queries_by_id,
    load_jsonl,
)


def _clip(s: str, n: int = 220) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def _print_verbose_demo(engine: BowSearchEngine, query: str, gold_docs: set[str]) -> None:
    analyzer = engine.vectorizer.build_analyzer()
    toks = list(analyzer(query))
    print(f"  token_preview ({len(toks)} tokens): {toks[:35]}{' ...' if len(toks) > 35 else ''}")
    q_row = engine.vectorizer.transform([query])
    qn = normalize(q_row, norm="l2", axis=1)
    q_dense = np.asarray(qn.toarray()).ravel()
    nz = int(np.sum(q_dense != 0))
    print(f"  query_vector: length={q_dense.shape[0]}, nonzeros={nz} (L2-normalized raw counts)")
    names = engine.vectorizer.get_feature_names_out()
    top_idx = np.argsort(-q_dense)[:12]
    pairs = [(names[i], float(q_dense[i])) for i in top_idx if q_dense[i] != 0]
    if pairs:
        print("  largest_query_dimensions (term, weight):")
        for t, w in pairs:
            print(f"    {t!r}: {w:.5f}")
    scores = engine.rank(query)
    order = np.argsort(-scores)
    print("  top_5_retrieved_passages (rank, score, snippet):")
    for r, idx in enumerate(order[:5]):
        doc_txt = engine.corpus[int(idx)]
        mark = "  <-- gold" if doc_txt in gold_docs else ""
        print(f"    #{r}  score={scores[int(idx)]:.5f}  {_clip(doc_txt, 160)}{mark}")
    gold_ranks = []
    for d in gold_docs:
        if d not in engine._text_to_index:
            continue
        g = engine._text_to_index[d]
        s = scores[g]
        gr = int(np.sum(scores > s) + np.sum((scores == s) & (np.arange(len(scores)) < g)))
        gold_ranks.append((gr, g, _clip(d, 120)))
    if gold_ranks:
        gold_ranks.sort(key=lambda x: x[0])
        print("  gold_passage_ranks (best rank among labeled positives):")
        for gr, _gi, snip in gold_ranks[:3]:
            print(f"    rank={gr}  {_clip(snip, 160)}")
        if len(gold_ranks) > 3:
            print(f"    ... ({len(gold_ranks) - 3} more gold variants omitted)")


def main() -> None:
    root = _ROOT
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, default=root / "data" / "msmarco_pairs")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 50, 100])
    p.add_argument("--max_features", type=int, default=65_536)
    p.add_argument("--ngram_min", type=int, default=1)
    p.add_argument("--ngram_max", type=int, default=1)
    p.add_argument("--binary", action="store_true", help="Binary bag-of-words (presence) instead of raw counts.")
    p.add_argument("--limit_queries", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    verbose = not args.quiet
    d = args.data_dir
    train_p = d / "train.jsonl"
    val_p = d / "validation.jsonl"
    test_p = d / "test.jsonl"
    for path in (train_p, val_p, test_p):
        if not path.exists():
            raise FileNotFoundError(path)
    if verbose:
        print("=== Bag-of-words cosine baseline walkthrough ===")
        print(f"project_root: {root}")
        print(f"reading_splits: {train_p.name}, {val_p.name}, {test_p.name}")
        sample_rows = load_jsonl(train_p)[:2]
        print("sample_raw_rows (from train.jsonl), as Python dicts:")
        for i, row in enumerate(sample_rows):
            preview = {k: row[k] for k in ("query_id", "query", "split")}
            preview["document"] = _clip(row["document"], 180)
            print(f"  record_{i+1}: {json.dumps(preview, ensure_ascii=False)}")
        print(
            "corpus_build: scanning splits in order, keeping first occurrence of each "
            "unique document string (deduplicated passage list for the index)."
        )
    corpus = build_unique_corpus([train_p, val_p, test_p])
    if verbose:
        print(f"unique_documents_in_index: {len(corpus)}")
        print(f"example_indexed_passage[0]: {_clip(corpus[0], 260)}")
    engine = BowSearchEngine.fit(
        corpus,
        max_features=args.max_features,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        binary=args.binary,
    )
    dm = engine.doc_matrix_norm
    nnz = getattr(dm, "nnz", None)
    voc = len(engine.vectorizer.vocabulary_)
    if verbose:
        print("=== vectorizer (sklearn CountVectorizer + row L2 normalize) ===")
        print(
            f"  lowercase=True  binary={args.binary}  dtype=float32  "
            f"ngram_range=({args.ngram_min}, {args.ngram_max})  max_features={args.max_features}"
        )
        print(
            "  each row is L2-normalized bag-of-words; dot product between query and doc rows "
            "equals cosine similarity between raw count vectors."
        )
        print(f"  learned vocabulary size (columns): {voc}")
        print(f"  document_matrix shape: {dm.shape[0]} docs x {dm.shape[1]} terms")
        if nnz is not None:
            total = max(dm.shape[0] * dm.shape[1], 1)
            print(f"  matrix nonzeros: {nnz} (~{100.0 * nnz / total:.4f}% dense if counted naively)")
        feats = engine.vectorizer.get_feature_names_out()
        print(f"  sample_vocabulary_terms: {list(feats[:18])}{' ...' if len(feats) > 18 else ''}")
        print("=== scoring model ===")
        print(
            "  for each query: CountVectorizer transform, L2-normalize, "
            "similarity_row = query_row @ document_matrix.T (cosine on BoW counts)."
        )
    if verbose:
        print("bow_fit_done")
    test_rows = load_jsonl(test_p)
    gold, qtext = group_queries_by_id(test_rows)
    qids = sorted(gold.keys())
    if args.limit_queries is not None:
        qids = qids[: args.limit_queries]
    if verbose and qids:
        demo_qid = qids[0]
        dq = qtext[demo_qid]
        dg = gold[demo_qid]
        print("=== one concrete test query (first query_id in eval order) ===")
        print(f"  query_id={demo_qid}")
        print(f"  query_text: {dq!r}")
        _print_verbose_demo(engine, dq, dg)
    metrics = evaluate_ranking(engine, gold, qtext, args.ks, query_ids=qids)
    if verbose:
        print("=== metrics (full eval set or limited by --limit_queries) ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
