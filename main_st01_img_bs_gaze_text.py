#!/usr/bin/env python3
"""Wrapper to train ST01 model with pretrained image+bbox+seg+gaze+text for 3-class classification."""

from __future__ import annotations

import sys

from main_st01_prtr_img_bs_gaze_text import main as base_main


def main() -> None:
    args = list(sys.argv)
    # Ensure bbox/seg/text/gaze are enabled by removing disabling flags
    for flag in ["--no-bbox", "--no-seg", "--no-text", "--no-gaze"]:
        while flag in args:
            args.remove(flag)
    # Keep image disabled since DICOM files are not available
    # Ensure pretrained image encoder is disabled
    while "--pretrained-image" in args:
        args.remove("--pretrained-image")
    sys.argv = args
    base_main()


if __name__ == "__main__":
    main()
