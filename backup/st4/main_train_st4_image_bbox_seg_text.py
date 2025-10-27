#!/usr/bin/env python3
"""Wrapper to train ST4 model with image+bbox+seg+text+gaze and imbalance correction."""

from __future__ import annotations

import sys

from main_train_st4 import main as base_main


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
    # Ensure imbalance correction and threshold tuning stay enabled
    while "--no-pos-weight" in args:
        args.remove("--no-pos-weight")
    while "--no-weighted-sampler" in args:
        args.remove("--no-weighted-sampler")
    while "--no-tune-thresholds" in args:
        args.remove("--no-tune-thresholds")
    sys.argv = args
    base_main()


if __name__ == "__main__":
    main()
