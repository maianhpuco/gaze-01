#!/usr/bin/env python3
"""Wrapper to train ST1 pretrained model with image+bbox+seg (no text)."""

from __future__ import annotations

import sys

from main_train_st01_pretrained import main as base_main


def main() -> None:
    args = sys.argv
    # Remove any disabling flags for bbox/seg/image if present
    for flag in ["--no-bbox", "--no-seg", "--no-image"]:
        while flag in args:
            args.remove(flag)
    # Ensure text decoding stays off
    if "--no-text" not in args:
        args.append("--no-text")
    base_main()


if __name__ == "__main__":
    main()
