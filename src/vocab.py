"""Simple word-level vocabulary and tokenizer for the LSTM encoder."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import torch


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1


def tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric characters."""
    return re.findall(r"[a-z0-9]+", text.lower())


class Vocabulary:
    """Word-to-index mapping with fixed special tokens at positions 0 and 1."""

    def __init__(self, word2idx: dict[str, int]) -> None:
        self.word2idx = word2idx
        self.idx2word = {i: w for w, i in word2idx.items()}
        self.pad_id = PAD_ID
        self.unk_id = UNK_ID

    def __len__(self) -> int:
        return len(self.word2idx)

    def __contains__(self, word: str) -> bool:
        return word in self.word2idx

    @classmethod
    def build(cls, texts: list[str], max_vocab: int = 50_000, min_freq: int = 2) -> "Vocabulary":
        """Build vocabulary from a list of raw text strings.

        Counts every word across all texts, keeps the top *max_vocab* words
        that appear at least *min_freq* times.
        """
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(tokenize(text))

        word2idx: dict[str, int] = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
        next_id = 2
        for word, freq in counts.most_common():
            if freq < min_freq:
                break
            if next_id >= max_vocab + 2:
                break
            word2idx[word] = next_id
            next_id += 1

        return cls(word2idx)

    def encode_text(self, text: str) -> list[int]:
        """Convert a raw string to a list of token IDs."""
        return [self.word2idx.get(w, self.unk_id) for w in tokenize(text)]

    def encode_batch(
        self,
        texts: list[str],
        max_length: int,
    ) -> dict[str, torch.Tensor]:
        """Tokenize, truncate, pad a batch of strings.

        Returns a dict with ``input_ids`` and ``attention_mask`` tensors,
        matching the interface that the rest of the codebase expects.
        """
        batch_ids: list[list[int]] = []
        batch_mask: list[list[int]] = []

        for text in texts:
            ids = self.encode_text(text)[:max_length]
            mask = [1] * len(ids)
            pad_len = max_length - len(ids)
            ids += [self.pad_id] * pad_len
            mask += [0] * pad_len
            batch_ids.append(ids)
            batch_mask.append(mask)

        return {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
        }

    def save(self, path: Path) -> None:
        """Save vocabulary to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "max_vocab": len(self.word2idx) - 2,
            "word2idx": self.word2idx,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        """Load vocabulary from a previously saved JSON file."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        word2idx = {w: int(i) for w, i in raw["word2idx"].items()}
        return cls(word2idx)
