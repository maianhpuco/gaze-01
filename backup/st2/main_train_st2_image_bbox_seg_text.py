#!/usr/bin/env python3
"""Wrapper to train ST2 model with image+bbox+seg+text+gaze."""

from __future__ import annotations

import sys

from st2.main_train_st2 import main as base_main


def main() -> None:
    args = list(sys.argv)
    # Ensure bbox/seg/image are enabled by removing disabling flags
    for flag in ["--no-bbox", "--no-seg", "--no-image"]:
        while flag in args:
            args.remove(flag)
    # Ensure text is enabled
    while "--no-text" in args:
        args.remove("--no-text")
    # Ensure gaze is enabled
    while "--no-gaze" in args:
        args.remove("--no-gaze")
    sys.argv = args
    base_main()


if __name__ == "__main__":
    main()
