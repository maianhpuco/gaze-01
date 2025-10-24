#!/usr/bin/env python3
"""Ablation training script: remove bbox, segmentation, image features, and transcript decoder."""

from __future__ import annotations

import sys
from typing import List

from main_train_silence_thought import main as base_main


def main() -> None:
    overrides: List[str] = [
        "--config",
        "config_maui/st_edg_cxr_rm_bbox_seg_img_text.yaml",
        "--no-bbox",
        "--no-seg",
        "--no-image",
        "--no-text",
    ]
    sys.argv = [sys.argv[0]] + overrides + sys.argv[1:]
    base_main()


if __name__ == "__main__":
    main()
