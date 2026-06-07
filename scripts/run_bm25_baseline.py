"""BM25 baseline evaluation on MS MARCO — same protocol as TF-IDF and BoW scripts."""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from src.bm25_retrieval import BM25SearchEngine, _tokenize
from src.tfidf_retrieval import (
    build_unique_corpus,
    evaluate_ranking,
    group_queries_by_id,
    load_jsonl,
)


def _clip(s: str, n: int = 220) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def _print_verbose_demo(engine: BM25SearchEngine, query: str, gold_docs: set[str]) -> None:
    toks = _tokenize(query)
    print(f"  token_preview ({len(toks)} tokens): {toks[:35]}{' ...' if len(toks) > 35 else ''}")
    scores = engine.rank(query)
    order = np.argsort(-scores)
    print("  top_5_retrieved_passages (rank, score, snippet):")
    for r, idx in enumerate(order[:5]):
        doc_txt = engine.corpus[int(idx)]
        mark = "  <-- gold" if doc_txt in gold_docs else ""
        print(f"    #{r}  score={scores[int(idx)]:.4f}  {_clip(doc_txt, 160)}{mark}")
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
    p = argparse.ArgumentParser(description="BM25Okapi baseline evaluation on MS MARCO.")
    p.add_argument("--data_dir", type=Path, default=root / "data" / "msmarco_pairs")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 50, 100])
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
        print("=== BM25 (Okapi) baseline ===")
        print(f"project_root: {root}")
        print(f"reading_splits: {train_p.name}, {val_p.name}, {test_p.name}")
        sample_rows = load_jsonl(train_p)[:2]
        print("sample_raw_rows (from train.jsonl):")
        for i, row in enumerate(sample_rows):
            preview = {k: row[k] for k in ("query_id", "query", "split")}
            preview["document"] = _clip(row["document"], 180)
            print(f"  record_{i+1}: {json.dumps(preview, ensure_ascii=False)}")
        print(
            "corpus_build: scanning splits in order, keeping first occurrence of each "
            "unique document string."
        )

    corpus = build_unique_corpus([train_p, val_p, test_p])
    if verbose:
        print(f"unique_documents_in_index: {len(corpus)}")
        print(f"example_indexed_passage[0]: {_clip(corpus[0], 260)}")
        print("fitting BM25Okapi (tokenise + IDF + term frequency saturation)...")

    engine = BM25SearchEngine.fit(corpus)

    if verbose:
        print("=== BM25Okapi parameters ===")
        print(f"  k1={engine.bm25.k1}  b={engine.bm25.b}  epsilon={engine.bm25.epsilon}")
        print(f"  corpus_size: {engine.bm25.corpus_size} docs")
        print(f"  avgdl: {engine.bm25.avgdl:.1f} tokens")
        print("=== scoring model ===")
        print(
            "  for each query: tokenise → BM25 term scores → sum over query terms → "
            "one score per document, then sort descending."
        )
        print("bm25_fit_done")

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
        print("=== metrics ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
