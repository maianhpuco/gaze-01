#!/usr/bin/env python3
"""Wrapper to train the pretrained ST1 model without bbox/seg features."""

from __future__ import annotations

import sys

from main_train_st01_pretrained import main as base_main


def main() -> None:
    if "--no-bbox" not in sys.argv and "--use-bbox" not in sys.argv:
        sys.argv.append("--no-bbox")
    if "--no-seg" not in sys.argv and "--use-seg" not in sys.argv:
        sys.argv.append("--no-seg")
    base_main()


if __name__ == "__main__":
    main()

