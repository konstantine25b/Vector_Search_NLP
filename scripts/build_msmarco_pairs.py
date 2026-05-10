import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def extract_pairs(example: dict, split_name: str) -> list[dict]:
    pairs: list[dict] = []
    qid = example["query_id"]
    query = example["query"]
    texts = example["passages"]["passage_text"]
    selected = example["passages"]["is_selected"]
    for passage_text, is_sel in zip(texts, selected):
        if is_sel:
            pairs.append(
                {
                    "query_id": qid,
                    "query": query,
                    "document": passage_text,
                    "split": split_name,
                }
            )
    return pairs


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_split(split_name: str, hf_split: str, max_rows: int | None) -> tuple[list[dict], int, int]:
    if max_rows is not None:
        ds = load_dataset("microsoft/ms_marco", "v1.1", split=f"{hf_split}[:{max_rows}]")
    else:
        ds = load_dataset("microsoft/ms_marco", "v1.1", split=hf_split)
    all_pairs: list[dict] = []
    skipped = 0
    for ex in tqdm(ds, desc=f"{split_name}"):
        pairs = extract_pairs(ex, split_name)
        if not pairs:
            skipped += 1
            continue
        all_pairs.extend(pairs)
    return all_pairs, len(ds), skipped


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "msmarco_pairs",
    )
    p.add_argument("--max_train", type=int, default=None)
    p.add_argument("--max_validation", type=int, default=None)
    p.add_argument("--max_test", type=int, default=None)
    p.add_argument("--sample_path", type=Path, default=None)
    p.add_argument("--sample_lines", type=int, default=80)
    args = p.parse_args()

    out_dir = args.out_dir
    manifest: dict = {"dataset": "microsoft/ms_marco", "config": "v1.1", "splits": {}}

    for split_name, hf_split, max_key in [
        ("train", "train", args.max_train),
        ("validation", "validation", args.max_validation),
        ("test", "test", args.max_test),
    ]:
        rows, n_raw, n_skipped = build_split(split_name, hf_split, max_key)
        out_path = out_dir / f"{split_name}.jsonl"
        write_jsonl(rows, out_path)
        proj_root = Path(__file__).resolve().parents[1]
        try:
            rel_out = str(out_path.relative_to(proj_root))
        except ValueError:
            rel_out = str(out_path)
        manifest["splits"][split_name] = {
            "hf_split": hf_split,
            "rows_with_labels": n_raw,
            "queries_without_positive": n_skipped,
            "query_document_pairs": len(rows),
            "file": rel_out,
        }

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    sample_out = args.sample_path or (out_dir / "train.sample.jsonl")
    train_file = out_dir / "train.jsonl"
    if train_file.exists():
        with train_file.open("r", encoding="utf-8") as f_in, sample_out.open("w", encoding="utf-8") as f_out:
            for i, line in enumerate(f_in):
                if i >= args.sample_lines:
                    break
                f_out.write(line)


if __name__ == "__main__":
    main()
