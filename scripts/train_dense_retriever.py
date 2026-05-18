import argparse
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed as hf_set_seed

from src.biencoder import BiEncoder
from src.infonce_loss import infonce_loss
from src.tfidf_retrieval import load_jsonl
from src.torch_device import pick_torch_device


class QueryDocDataset(Dataset):
    def __init__(self, rows: list[tuple[str, str]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.rows[idx]


def collate_pairs(
    batch: list[tuple[str, str]],
    tokenizer: AutoTokenizer,
    max_query_length: int,
    max_doc_length: int,
) -> dict[str, dict[str, torch.Tensor]]:
    queries, docs = zip(*batch)
    q_enc = tokenizer(
        list(queries),
        padding=True,
        truncation=True,
        max_length=max_query_length,
        return_tensors="pt",
    )
    d_enc = tokenizer(
        list(docs),
        padding=True,
        truncation=True,
        max_length=max_doc_length,
        return_tensors="pt",
    )
    return {"query": q_enc, "document": d_enc}


def load_query_doc_rows(path: Path, limit: int | None) -> list[tuple[str, str]]:
    raw = load_jsonl(path)
    rows = [(str(r["query"]), str(r["document"])) for r in raw]
    if limit is not None:
        rows = rows[:limit]
    return rows


def _make_loader(
    tokenizer: AutoTokenizer,
    pairs: list[tuple[str, str]],
    batch_size: int,
    shuffle: bool,
    max_query_length: int,
    max_doc_length: int,
    num_workers: int,
    seed: int,
) -> DataLoader:

    ds = QueryDocDataset(pairs)

    def _collate(b: list[tuple[str, str]]) -> dict[str, dict[str, torch.Tensor]]:
        return collate_pairs(b, tokenizer, max_query_length, max_doc_length)

    g = None
    if shuffle and num_workers > 0:
        g = torch.Generator()
        g.manual_seed(seed)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=g,
        collate_fn=_collate,
        drop_last=False,
    )


@torch.no_grad()
def estimate_loss(
    model: BiEncoder,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
    symmetric: bool,
    max_batches: int | None = None,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        qb = batch["query"]
        db = batch["document"]
        q_emb, d_emb = model(
            qb["input_ids"].to(device),
            qb["attention_mask"].to(device),
            db["input_ids"].to(device),
            db["attention_mask"].to(device),
        )
        loss = infonce_loss(q_emb, d_emb, temperature=temperature, symmetric=symmetric)
        bsz = q_emb.shape[0]
        total += float(loss.item()) * bsz
        n += bsz
    model.train()
    return total / max(n, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train bi-encoder with MS MARCO (query, passage) pairs.")
    p.add_argument(
        "--train_jsonl",
        type=Path,
        default=_ROOT / "data" / "msmarco_pairs" / "train.jsonl",
        help="Training JSONL (query/document fields).",
    )
    p.add_argument(
        "--validation_jsonl",
        type=Path,
        default=_ROOT / "data" / "msmarco_pairs" / "validation.jsonl",
        help="Optional validation JSONL for loss logging.",
    )
    p.add_argument("--out_dir", type=Path, default=_ROOT / "checkpoints" / "dense_msmarco")
    p.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_query_tokens", type=int, default=64)
    p.add_argument("--max_doc_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--symmetric_infonce", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pooling", type=str, default="mean", choices=("mean", "cls"))
    p.add_argument(
        "--proj_dim",
        type=int,
        default=None,
        help="If set, add a linear head to this size before L2 normalize (default: backbone hidden size).",
    )
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_val_batches", type=int, default=100, help="Cap validation batches per check (speed).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", help="auto (cuda > mps > cpu) | cpu | mps | cuda | cuda:0 ...")
    p.add_argument("--save_every_epoch", action="store_true")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    hf_set_seed(args.seed)

    device = pick_torch_device(args.device)
    train_rows = load_query_doc_rows(args.train_jsonl, args.max_train_samples)
    if not train_rows:
        raise RuntimeError(f"No training rows read from {args.train_jsonl}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = BiEncoder(args.model_name, pooling=args.pooling, proj_dim=args.proj_dim).to(device)
    opt = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train_loader = _make_loader(
        tokenizer,
        train_rows,
        batch_size=args.batch_size,
        shuffle=True,
        max_query_length=args.max_query_tokens,
        max_doc_length=args.max_doc_tokens,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    val_loader_full: DataLoader | None = None
    if args.validation_jsonl.exists():
        val_rows = load_query_doc_rows(args.validation_jsonl, limit=None)
        val_loader_full = _make_loader(
            tokenizer,
            val_rows,
            batch_size=args.batch_size,
            shuffle=False,
            max_query_length=args.max_query_tokens,
            max_doc_length=args.max_doc_tokens,
            num_workers=args.num_workers,
            seed=args.seed,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "model_name": args.model_name,
        "pooling": args.pooling,
        "proj_dim": args.proj_dim,
        "max_query_tokens": args.max_query_tokens,
        "max_doc_tokens": args.max_doc_tokens,
        "temperature": args.temperature,
        "symmetric_infonce": args.symmetric_infonce,
        "train_jsonl": str(args.train_jsonl),
        "n_train_pairs": len(train_rows),
    }
    with (args.out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in pbar:
            qb = batch["query"]
            db = batch["document"]
            q_emb, d_emb = model(
                qb["input_ids"].to(device),
                qb["attention_mask"].to(device),
                db["input_ids"].to(device),
                db["attention_mask"].to(device),
            )
            loss = infonce_loss(
                q_emb,
                d_emb,
                temperature=args.temperature,
                symmetric=args.symmetric_infonce,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            global_step += 1
            pbar.set_postfix(loss=f"{float(loss.item()):.4f}")

        last_path = args.out_dir / "last.pt"
        torch.save(
            {
                "encoder_state_dict": model.state_dict(),
                "model_name": args.model_name,
                "pooling": args.pooling,
                "proj_dim": args.proj_dim,
                "global_step": global_step,
                "epoch": epoch,
                "tokenizer_name": args.model_name,
                "max_query_tokens": args.max_query_tokens,
                "max_doc_tokens": args.max_doc_tokens,
                "temperature": args.temperature,
                "symmetric_infonce": args.symmetric_infonce,
            },
            last_path,
        )

        val_loss: float | None = None
        if val_loader_full is not None:
            val_loss = estimate_loss(
                model,
                val_loader_full,
                device,
                args.temperature,
                args.symmetric_infonce,
                max_batches=args.max_val_batches if args.max_val_batches > 0 else None,
            )
            tqdm.write(f"[epoch {epoch}] validation_approx_loss={val_loss:.4f}")

        if args.save_every_epoch:
            ep_path = args.out_dir / f"epoch{epoch}.pt"
            torch.save(
                {
                    "encoder_state_dict": model.state_dict(),
                    "model_name": args.model_name,
                    "pooling": args.pooling,
                    "proj_dim": args.proj_dim,
                    "global_step": global_step,
                    "epoch": epoch,
                    "tokenizer_name": args.model_name,
                    "max_query_tokens": args.max_query_tokens,
                    "max_doc_tokens": args.max_doc_tokens,
                    "val_approx_loss": val_loss,
                    "temperature": args.temperature,
                    "symmetric_infonce": args.symmetric_infonce,
                },
                ep_path,
            )

    tqdm.write(f"checkpoint_saved: {args.out_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
