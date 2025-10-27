#!/usr/bin/env python3
"""Wrapper to train ST1 pretrained model with bbox+seg+gaze+text (no image)."""

from __future__ import annotations

import sys

from main_train_st01_pretrained import main as base_main


def main() -> None:
    args = sys.argv
    # Ensure bbox/seg are enabled (remove disabling flags)
    for flag in ["--no-bbox", "--no-seg"]:
        while flag in args:
            args.remove(flag)
    # Disable image branch
    if "--no-image" not in args:
        args.append("--no-image")
    # Ensure text is enabled
    while "--no-text" in args:
        args.remove("--no-text")
    base_main()


if __name__ == "__main__":
    main()

