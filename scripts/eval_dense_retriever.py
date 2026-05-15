import argparse
import hashlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from transformers import AutoTokenizer

from src.biencoder import BiEncoder
from src.dense_retrieval import DenseSearchEngine, encode_corpus
from src.tfidf_retrieval import build_unique_corpus, evaluate_ranking, group_queries_by_id, load_jsonl
from src.torch_device import pick_torch_device


def corpus_fingerprint(jsonl_paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in jsonl_paths:
        st = p.stat()
        h.update(str(p.resolve()).encode())
        h.update(str(st.st_size).encode())
        h.update(str(st.st_mtime_ns).encode())
    return h.hexdigest()[:24]


def checkpoint_identity(path: Path) -> tuple[int, int]:
    st = path.stat()
    return (st.st_size, st.st_mtime_ns)


def default_doc_cache_npz(checkpoint: Path, corpus_fp: str, max_doc_tokens: int, pooling: str, proj_raw) -> Path:
    proj = proj_raw if proj_raw is not None else "none"
    key = f"{checkpoint.stem}__corp{corpus_fp}__m{max_doc_tokens}__pool{pooling}__proj{proj}"
    d = checkpoint.parent / "doc_emb_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.npz"


def default_doc_cache_meta(cache_npz: Path) -> Path:
    return cache_npz.with_suffix(".meta.json")


def load_doc_emb_cache(meta_path: Path, npz_path: Path) -> np.ndarray | None:
    if not meta_path.exists() or not npz_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("schema") != 1:
        return None
    data = np.load(npz_path)
    emb_f16 = data["emb"].astype(np.float32)
    n_docs = int(meta["n_docs"])
    dim = int(meta["embedding_dim"])
    if emb_f16.shape != (n_docs, dim):
        return None
    return emb_f16


def write_doc_emb_cache(
    npz_path: Path,
    meta_path: Path,
    doc_emb: np.ndarray,
    *,
    corpus_fp: str,
    checkpoint: Path,
    ckpt_ident: tuple[int, int],
    max_doc_tokens: int,
    model_name: str,
    pooling: str,
    proj_dim: int | None,
) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, emb=doc_emb.astype(np.float16))
    meta = {
        "schema": 1,
        "corpus_fingerprint": corpus_fp,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_identity": {"size": ckpt_ident[0], "mtime_ns": ckpt_ident[1]},
        "max_doc_tokens": max_doc_tokens,
        "model_name": model_name,
        "pooling": pooling,
        "proj_dim": proj_dim,
        "n_docs": doc_emb.shape[0],
        "embedding_dim": doc_emb.shape[1],
        "emb_dtype_saved": "float16",
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def meta_matches(
    meta_path: Path,
    *,
    corpus_fp: str,
    checkpoint: Path,
    ckpt_ident: tuple[int, int],
    max_doc_tokens: int,
    model_name: str,
    pooling: str,
    proj_dim: int | None,
    n_docs: int,
) -> bool:
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if meta.get("schema") != 1:
        return False
    ident = meta.get("checkpoint_identity") or {}
    return (
        meta.get("corpus_fingerprint") == corpus_fp
        and meta.get("checkpoint_path") == str(checkpoint.resolve())
        and int(ident.get("size", -1)) == ckpt_ident[0]
        and int(ident.get("mtime_ns", -1)) == ckpt_ident[1]
        and int(meta.get("max_doc_tokens", -1)) == max_doc_tokens
        and meta.get("model_name") == model_name
        and meta.get("pooling") == pooling
        and (meta.get("proj_dim") == proj_dim)
        and int(meta.get("n_docs", -1)) == n_docs
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained bi-encoder on the MS MARCO JSONL splits.")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=_ROOT / "checkpoints" / "dense_msmarco" / "last.pt",
    )
    p.add_argument("--data_dir", type=Path, default=_ROOT / "data" / "msmarco_pairs")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 50, 100])
    p.add_argument(
        "--limit_queries",
        type=int,
        default=None,
        help="Evaluate only this many DISTINCT test queries (after building the full retrieval index). "
        "Does NOT skip encoding passages: slow work is indexing ~all unique passages once; use "
        "doc_emb_cache on the second run, or reduce data via msmarco build --max_* for a toy corpus.",
    )
    p.add_argument(
        "--encode_batch_size",
        type=int,
        default=24,
        help="Passage encoding batch size. Lower (e.g. 8–16) if MPS/CPU runs out of memory.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto (cuda > mps > cpu) | cpu | mps | cuda | cuda:0 ...",
    )
    p.add_argument("--max_query_tokens", type=int, default=None, help="Override checkpoint default if set.")
    p.add_argument("--max_doc_tokens", type=int, default=None, help="Override checkpoint default if set.")
    p.add_argument(
        "--doc_emb_cache",
        type=Path,
        default=None,
        help="Override path for cached document embeddings (.npz). Default: next to checkpoint under doc_emb_cache/.",
    )
    p.add_argument(
        "--no_doc_emb_cache",
        action="store_true",
        help="Always encode the full corpus; do not read or write cache files.",
    )
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_p = args.data_dir / "train.jsonl"
    val_p = args.data_dir / "validation.jsonl"
    test_p = args.data_dir / "test.jsonl"
    split_paths = [train_p, val_p, test_p]
    for path in split_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    ckpt_ident = checkpoint_identity(args.checkpoint)
    corp_fp = corpus_fingerprint(split_paths)

    device = pick_torch_device(args.device)
    # Load checkpoint on CPU first (recommended for MPS / portability).
    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")

    model_name = ckpt["model_name"]
    pooling = ckpt.get("pooling", "mean")
    proj_dim = ckpt.get("proj_dim")
    max_q = args.max_query_tokens if args.max_query_tokens is not None else ckpt.get("max_query_tokens", 64)
    max_d = args.max_doc_tokens if args.max_doc_tokens is not None else ckpt.get("max_doc_tokens", 256)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BiEncoder(model_name, pooling=pooling, proj_dim=proj_dim)
    model.load_state_dict(ckpt["encoder_state_dict"])
    model.to(device)
    model.eval()

    corpus = build_unique_corpus(split_paths)

    cache_npz = args.doc_emb_cache
    meta_path = default_doc_cache_meta(cache_npz) if cache_npz is not None else None
    if cache_npz is None and not args.no_doc_emb_cache:
        cache_npz = default_doc_cache_npz(args.checkpoint, corp_fp, max_d, pooling, proj_dim)
        meta_path = default_doc_cache_meta(cache_npz)

    if not args.quiet:
        print(f"unique_documents_in_index: {len(corpus)}")
        print(f"device={device} checkpoint={args.checkpoint}")
        if args.limit_queries is not None:
            print(
                f"NOTE: --limit_queries={args.limit_queries} skips extra query scoring only. "
                f"The slow step is encoding all {len(corpus)} passages once (until doc_emb_cache hits)."
            )
    doc_emb = None
    if not args.no_doc_emb_cache and cache_npz is not None and meta_path is not None:
        if meta_matches(
            meta_path,
            corpus_fp=corp_fp,
            checkpoint=args.checkpoint,
            ckpt_ident=ckpt_ident,
            max_doc_tokens=int(max_d),
            model_name=model_name,
            pooling=pooling,
            proj_dim=proj_dim,
            n_docs=len(corpus),
        ):
            doc_emb = load_doc_emb_cache(meta_path, cache_npz)
            if doc_emb is not None and not args.quiet:
                print(f"loaded_doc_emb_cache: {cache_npz}")

    if doc_emb is None:
        if not args.quiet:
            hint = ""
            if not args.no_doc_emb_cache and cache_npz is not None:
                hint = f" (will save cache to {cache_npz})"
            print(f"encoding_full_corpus...{hint}")
        doc_emb = encode_corpus(
            model,
            tokenizer,
            corpus,
            device,
            batch_size=args.encode_batch_size,
            max_length=int(max_d),
            show_progress=not args.quiet,
        )
        if not args.no_doc_emb_cache and cache_npz is not None and meta_path is not None:
            write_doc_emb_cache(
                cache_npz,
                meta_path,
                doc_emb,
                corpus_fp=corp_fp,
                checkpoint=args.checkpoint,
                ckpt_ident=ckpt_ident,
                max_doc_tokens=int(max_d),
                model_name=model_name,
                pooling=pooling,
                proj_dim=proj_dim,
            )
            if not args.quiet:
                print(f"wrote_doc_emb_cache: {cache_npz}")

    engine = DenseSearchEngine(
        tokenizer,
        model,
        corpus,
        doc_emb,
        device,
        max_query_length=int(max_q),
    )

    test_rows = load_jsonl(test_p)
    gold, qtext = group_queries_by_id(test_rows)
    qids = sorted(gold.keys())
    if args.limit_queries is not None:
        qids = qids[: args.limit_queries]

    if not args.quiet and args.limit_queries is not None:
        print(f"scoring_queries: {len(qids)} (subset)")

    metrics = evaluate_ranking(engine, gold, qtext, args.ks, query_ids=qids)
    if not args.quiet:
        print("=== dense retriever metrics ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
