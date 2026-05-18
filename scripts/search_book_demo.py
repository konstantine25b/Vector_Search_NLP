import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(_ROOT / ".cache" / "huggingface"))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from src.tfidf_retrieval import TfidfSearchEngine, load_jsonl


def clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3].rstrip() + "..."


def load_chunks(path: Path) -> list[dict]:
    rows = load_jsonl(path)
    if not rows:
        raise RuntimeError(f"No chunks found in {path}. Run scripts/build_book_chunks.py first.")
    for row in rows:
        if "text" not in row:
            raise ValueError(f"Chunk row missing 'text': {row}")
    return rows


def print_hits(rows: list[dict], scores: np.ndarray, top_k: int, snippet_chars: int) -> None:
    order = np.argsort(-scores)[:top_k]
    for rank, idx in enumerate(order, start=1):
        row = rows[int(idx)]
        page_span = f"p.{row['page_start']}" if row["page_start"] == row["page_end"] else f"pp.{row['page_start']}-{row['page_end']}"
        print(f"\n#{rank}  score={float(scores[int(idx)]):.4f}  {row['chunk_id']}  {page_span}  words={row['n_words']}")
        print(clip(row["text"], snippet_chars))


def run_tfidf(rows: list[dict], query: str, top_k: int, max_features: int, ngram_max: int, snippet_chars: int) -> None:
    corpus = [r["text"] for r in rows]
    engine = TfidfSearchEngine.fit(
        corpus,
        max_features=max_features,
        ngram_min=1,
        ngram_max=ngram_max,
    )
    scores = engine.rank(query)
    print(f"method=tfidf chunks={len(rows)} max_features={max_features} ngram_range=(1,{ngram_max})")
    print_hits(rows, scores, top_k, snippet_chars)


def run_dense(
    rows: list[dict],
    query: str,
    top_k: int,
    checkpoint: Path,
    device_name: str,
    encode_batch_size: int,
    snippet_chars: int,
) -> None:
    import torch
    from transformers import AutoTokenizer

    from src.biencoder import BiEncoder
    from src.dense_retrieval import DenseSearchEngine, encode_corpus
    from src.torch_device import pick_torch_device

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    try:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location="cpu")

    model_name = ckpt["model_name"]
    pooling = ckpt.get("pooling", "mean")
    proj_dim = ckpt.get("proj_dim")
    max_query_tokens = int(ckpt.get("max_query_tokens", 64))
    max_doc_tokens = int(ckpt.get("max_doc_tokens", 256))
    device = pick_torch_device(device_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BiEncoder(model_name, pooling=pooling, proj_dim=proj_dim)
    model.load_state_dict(ckpt["encoder_state_dict"])
    model.to(device)
    model.eval()

    corpus = [r["text"] for r in rows]
    doc_emb = encode_corpus(
        model,
        tokenizer,
        corpus,
        device,
        batch_size=encode_batch_size,
        max_length=max_doc_tokens,
    )
    engine = DenseSearchEngine(tokenizer, model, corpus, doc_emb, device, max_query_length=max_query_tokens)
    scores = engine.rank(query)
    print(f"method=dense chunks={len(rows)} checkpoint={checkpoint} device={device}")
    print_hits(rows, scores, top_k, snippet_chars)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Search Jurafsky & Martin SLP book chunks.")
    p.add_argument("query", type=str, nargs="?", default=None, help="Search query. If omitted, you will be prompted.")
    p.add_argument("--chunks", type=Path, default=_ROOT / "data" / "book_chunks" / "slp_chunks.jsonl")
    p.add_argument("--method", choices=("tfidf", "dense"), default="tfidf")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--max_features", type=int, default=65_536)
    p.add_argument("--ngram_max", type=int, default=1)
    p.add_argument("--checkpoint", type=Path, default=_ROOT / "checkpoints" / "dense_msmarco" / "last.pt")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--encode_batch_size", type=int, default=16)
    p.add_argument("--snippet_chars", type=int, default=900)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    query = args.query or input("query> ").strip()
    if not query:
        raise SystemExit("Empty query.")

    rows = load_chunks(args.chunks)
    print(f"query={query!r}")
    if args.method == "tfidf":
        run_tfidf(rows, query, args.top_k, args.max_features, args.ngram_max, args.snippet_chars)
    else:
        run_dense(rows, query, args.top_k, args.checkpoint, args.device, args.encode_batch_size, args.snippet_chars)


if __name__ == "__main__":
    main()
