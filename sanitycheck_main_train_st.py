#!/usr/bin/env python3
"""
Sanity checker for the gaze-intent RNN inputs.

Loads a single batch using the same configuration as training, prints the key tensor
shapes, and runs the new GazeSeqRNNAttend model to expose intermediate features
such as encoder inputs and speak-gate logits. This is intended for quick inspection
before launching long training runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from egd_cxr_dataset import ConfigLoader, EGDCXRDataset, create_dataloader  # noqa: E402
from main_train_st import (  # noqa: E402
    build_model_and_vocab,
    read_split_ids,
)


def _resolve_split_ids(config_loader: ConfigLoader, split: str) -> list[str]:
    split_dir_cfg = config_loader.get("split_files", "dir", default=ROOT / "config" / "splits")
    split_dir = Path(split_dir_cfg)
    if not split_dir.is_absolute():
        split_dir = ROOT / split_dir
    return read_split_ids(split_dir, split)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect inputs to the GazeSeqRNNAttend encoder.")
    parser.add_argument("--configs", type=Path, required=True, help="Path to the YAML configuration file.")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=("train", "val", "test"),
        help="Dataset split to sample from (default: train).",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Mini-batch size for inspection (default: 1).")
    parser.add_argument("--max-fixations", type=int, default=64, help="Maximum fixations per case (default: 64).")
    parser.add_argument("--enc-dim", type=int, default=256, help="Encoder/intent dimension (default: 256).")
    parser.add_argument("--txt-dim", type=int, default=256, help="Decoder/text dimension (default: 256).")
    parser.add_argument("--num-workers", type=int, default=0, help="Data loader workers (default: 0).")
    args = parser.parse_args()

    config_loader = ConfigLoader(args.configs)

    gaze_root = Path(config_loader.get("input_path", "gaze_raw"))
    seg_dir = Path(config_loader.get("input_path", "segmentation_dir"))
    transcripts_dir = Path(config_loader.get("input_path", "transcripts_dir", default=seg_dir))
    dicom_root = Path(config_loader.get("input_path", "dicom_raw"))

    split_ids = _resolve_split_ids(config_loader, args.split)
    if not split_ids:
        raise RuntimeError(f"No case IDs found for split '{args.split}'.")

    dataset = EGDCXRDataset(
        root=gaze_root,
        seg_path=seg_dir,
        transcripts_path=transcripts_dir,
        dicom_root=dicom_root,
        max_fixations=args.max_fixations,
        case_ids=split_ids,
    )

    print(f"Dataset loaded: {len(dataset)} cases | max_fixations={args.max_fixations}")

    loader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    batch = next(iter(loader))
    fix = batch["fixations"]
    transcripts = batch["transcripts"]
    images = batch["images"]

    case_idx = 0
    length = int(fix["lengths"][case_idx].item())
    print(f"\nCase index {case_idx} | dicom_id={batch['dicom_ids'][case_idx]} | sequence length={length}")
    print(f"  xy shape:        {tuple(fix['xy'][case_idx, :length].shape)}")
    print(f"  dwell shape:     {tuple(fix['dwell'][case_idx, :length].shape)}")
    print(f"  time shape:      {tuple(fix['time'][case_idx, :length].shape)}")
    print(f"  seg_hits shape:  {tuple(fix['seg_hits'][case_idx, :length].shape)}")
    print(f"  box_hits shape:  {tuple(fix['box_hits'][case_idx, :length].shape)}")

    device = torch.device("cpu")
    model, vocab = build_model_and_vocab(
        dataset,
        device,
        txt_dim=args.txt_dim,
        enc_dim=args.enc_dim,
        max_decode_len=64,
        use_bbox=True,
        use_seg=True,
        use_image=True,
        use_text=True,
    )
    model.eval()

    case = {
        "xy": fix["xy"][case_idx, :length].to(device),
        "dwell": fix["dwell"][case_idx, :length].to(device),
        "time": fix["time"][case_idx, :length].to(device),
        "seg_hits": fix["seg_hits"][case_idx, :length].to(device),
        "box_hits": fix["box_hits"][case_idx, :length].to(device),
    }
    transcript = transcripts[case_idx]
    image = images[case_idx].to(device)

    feat_no_txt = model._build_fixation_features(  # type: ignore[attr-defined]
        case["xy"], case["dwell"], case["time"], case["seg_hits"], case["box_hits"]
    )
    print(f"\nEncoder feature tensor (without text context): {tuple(feat_no_txt.shape)}")
    print(f"GRU input size: {model.enc_gru.input_size}")

    outputs = model.forward_case(
        fixations=case,
        transcript=transcript,
        encode_text_fn=lambda s: vocab.encode(s, add_eos=True),
        image_1chw=image,
    )

    h_seq = outputs["h_seq"]
    speak_logits = outputs["speak_logits"]
    print(f"Encoder hidden sequence shape: {tuple(h_seq.shape)}")
    print(f"Speak logits shape: {tuple(speak_logits.shape)}")
    if speak_logits.numel() > 0:
        probs = torch.sigmoid(speak_logits)
        head = min(10, probs.numel())
        print(f"  First {head} speak probabilities: {probs[:head].tolist()}")

    txt_logits = outputs["txt_logits_per_segment"]
    print(f"Segments in transcript: {len(transcript.get('segments', [])) if isinstance(transcript, dict) else 0}")
    print(f"Decoder windows returned: {len(txt_logits)}")
    if txt_logits:
        lengths = [tensor.size(0) for tensor in txt_logits]
        print(f"  Token logits shapes: {lengths}")


if __name__ == "__main__":
    main()
