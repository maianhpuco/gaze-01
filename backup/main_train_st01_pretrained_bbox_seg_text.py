#!/usr/bin/env python3
"""Wrapper to train ST1 pretrained model with bbox+seg+text (no image, no gaze)."""

from __future__ import annotations

import sys

from main_train_st01_pretrained import main as base_main


def main() -> None:
    args = list(sys.argv)
    # Ensure bbox/seg features remain enabled
    for flag in ["--no-bbox", "--no-seg"]:
        while flag in args:
            args.remove(flag)
    # Disable image and gaze branches explicitly
    if "--no-image" not in args:
        args.append("--no-image")
    if "--no-gaze" not in args:
        args.append("--no-gaze")
    # Ensure text decoding stays on
    while "--no-text" in args:
        args.remove("--no-text")
    sys.argv = args
    base_main()


if __name__ == "__main__":
    main()
