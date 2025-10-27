#!/usr/bin/env python3
"""Wrapper to train the ST1 pretrained model using image only (no gaze/text/bbox/seg)."""

from __future__ import annotations

import sys

from main_train_st01_pretrained import main as base_main


def main() -> None:
    flags = ["--no-bbox", "--no-seg", "--no-text"]
    present = set(sys.argv)
    for flag in flags:
        if flag not in present:
            sys.argv.append(flag)
    base_main()


if __name__ == "__main__":
    main()

