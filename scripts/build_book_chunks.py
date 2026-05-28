import argparse
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pypdf import PdfReader


def normalize_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"([A-Za-z])-\s+([a-z])", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def words_with_pages(pdf_path: Path, start_page: int, end_page: int | None) -> list[tuple[str, int]]:
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    first = max(start_page, 1)
    last = total_pages if end_page is None else min(end_page, total_pages)
    if first > last:
        raise ValueError(f"Invalid page range: start_page={start_page}, end_page={end_page}")

    out: list[tuple[str, int]] = []
    for page_no in range(first, last + 1):
        raw = reader.pages[page_no - 1].extract_text() or ""
        cleaned = normalize_text(raw)
        for word in cleaned.split():
            out.append((word, page_no))
    return out


def make_chunks(
    tokens: list[tuple[str, int]],
    chunk_words: int,
    stride_words: int,
) -> list[dict]:
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if stride_words <= 0:
        raise ValueError("stride_words must be positive")

    chunks: list[dict] = []
    for start in range(0, len(tokens), stride_words):
        window = tokens[start : start + chunk_words]
        if len(window) < max(50, chunk_words // 2):
            break
        words = [w for w, _p in window]
        pages = [p for _w, p in window]
        chunks.append(
            {
                "chunk_id": f"slp_{len(chunks):05d}",
                "text": " ".join(words),
                "word_start": start,
                "word_end": start + len(window),
                "page_start": min(pages),
                "page_end": max(pages),
                "n_words": len(words),
            }
        )
    return chunks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract 200-300 word searchable chunks from the SLP PDF.")
    p.add_argument("--pdf", type=Path, default=_ROOT / "ed3book_jan26.pdf")
    p.add_argument("--out", type=Path, default=_ROOT / "data" / "book_chunks" / "slp_chunks.jsonl")
    p.add_argument("--chunk_words", type=int, default=240)
    p.add_argument("--stride_words", type=int, default=200)
    p.add_argument(
        "--start_page",
        type=int,
        default=10,
        help="PDF page to start extracting from. Default skips cover/table-of-contents front matter.",
    )
    p.add_argument("--end_page", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.pdf.exists():
        raise FileNotFoundError(args.pdf)

    tokens = words_with_pages(args.pdf, args.start_page, args.end_page)
    chunks = make_chunks(tokens, args.chunk_words, args.stride_words)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "pdf": str(args.pdf),
                "out": str(args.out),
                "n_words": len(tokens),
                "n_chunks": len(chunks),
                "chunk_words": args.chunk_words,
                "stride_words": args.stride_words,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
