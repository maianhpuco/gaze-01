#!/usr/bin/env python3
"""Wrapper to train the pretrained ST1 model without transcript decoding."""

from __future__ import annotations

import sys

from main_train_st01_pretrained import main as base_main


def main() -> None:  # noqa: D401 - thin wrapper
    if "--no-text" not in sys.argv and "--use-text" not in sys.argv:
        sys.argv.append("--no-text")
    base_main()


if __name__ == "__main__":
    main()

