#!/usr/bin/env python3
"""Wrapper to train ST1 pretrained model with image+bbox+seg+text, disabling gaze features."""

from __future__ import annotations

import sys

from main_train_st01_pretrained import main as base_main


def main() -> None:
    args = list(sys.argv)
    # Ensure bbox/seg/image are enabled by removing disabling flags
    for flag in ["--no-bbox", "--no-seg", "--no-image"]:
        while flag in args:
            args.remove(flag)
    # Ensure text is enabled
    while "--no-text" in args:
        args.remove("--no-text")
    # Force gaze features off
    if "--no-gaze" not in args:
        args.append("--no-gaze")
    sys.argv = args
    base_main()


if __name__ == "__main__":
    main()
