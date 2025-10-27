#!/usr/bin/env python3
"""
Lightweight vocabulary utilities for transcript tokenisation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"


def tokenize(text: str) -> List[str]:
    """Whitespace + punctuation splitting with lowercase normalisation."""
    if not text:
        return []
    return [tok.lower() for tok in re.findall(r"[A-Za-z0-9%+\-_/\.]+", text)]


@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    @property
    def size(self) -> int:
        return len(self.itos)

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = True,
    ) -> torch.Tensor:
        tokens = tokenize(text)
        ids = [self.stoi.get(tok, self.unk_id) for tok in tokens]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        if not ids:
            ids = [self.eos_id] if add_eos else [self.unk_id]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: Iterable[int]) -> str:
        tokens: List[str] = []
        for idx in ids:
            tok = self.itos[idx] if 0 <= idx < len(self.itos) else UNK
            if tok == EOS:
                break
            if tok in (PAD, BOS):
                continue
            tokens.append(tok)
        return " ".join(tokens)


def build_vocab(
    texts: List[str],
    *,
    min_freq: int = 1,
    max_size: int = 30000,
) -> Vocab:
    """Create a vocabulary from a corpus of strings."""
    from collections import Counter

    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    specials = [PAD, BOS, EOS, UNK]
    itos: List[str] = []
    itos.extend(specials)

    for token, freq in counter.most_common():
        if freq < min_freq:
            break
        if token in specials:
            continue
        itos.append(token)
        if len(itos) >= max_size:
            break

    stoi = {tok: idx for idx, tok in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)


__all__ = [
    "Vocab",
    "build_vocab",
    "tokenize",
    "PAD",
    "BOS",
    "EOS",
    "UNK",
]

