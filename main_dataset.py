#!/usr/bin/env python3
"""Inspect EGDCXRDataset (original DICOM-based) and save an example to sample/.

Prints:
- Total samples in train/val/test
- One example with:
  - image tensor shape
  - number of fixations and dwell-time statistics
  - per-fixation 1-hot vectors for segmentation hits and bbox hits (first few shown)
  - transcript segments (begin/end/time if present) and text content

Saves:
- JSON dump of the example under ./sample/sample_000.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from egd_cxr_dataset import ConfigLoader  # type: ignore
from egd_cxr_dataset.datasets.egd_cxr import (  # type: ignore
    EGDCXRDataset,
    create_dataloader,
)


def read_split_ids(split_dir: Path, split: str) -> List[str]:
    path = split_dir / f"{split}_ids.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    ids: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if s and not s.startswith("#"):
            ids.append(s)
    return ids


def _make_json_serializable(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (int, float, str)) or obj is None:
        return obj
    return str(obj)


def inspect_example(sample: Dict[str, Any]) -> None:
    dicom_id = sample.get("dicom_id", "unknown")
    print(f"\nExample dicom_id: {dicom_id}")

    # Image
    img = sample.get("image")
    if isinstance(img, torch.Tensor):
        print(f"\nimage shape: {tuple(img.shape)}")
    # Metadata-derived dimensions if available
    meta = sample.get("meta", {}) if isinstance(sample.get("meta", {}), dict) else {}
    dicom_pairs = [
        ("dicom_height", "dicom_width"),
        ("dicom_rows", "dicom_cols"),
        ("original_height", "original_width"),
        ("height", "width"),
    ]
    dicom_h = dicom_w = None
    for hk, wk in dicom_pairs:
        if hk in meta and wk in meta:
            dicom_h, dicom_w = int(meta[hk]), int(meta[wk])
            break

    seg_h = meta.get("segmentation_height")
    seg_w = meta.get("segmentation_width")
    img_h_meta = meta.get("image_height")
    img_w_meta = meta.get("image_width")

    if dicom_h and dicom_w:
        print(f"\noriginal DICOM size (H,W): {dicom_h} x {dicom_w}")
    if seg_h and seg_w:
        print(f"\nsegmentation grid size (H,W): {seg_h} x {seg_w}")
    if img_h_meta and img_w_meta:
        print(f"\nreported tensor size (H,W): {img_h_meta} x {img_w_meta}")

    # Fixations (handle batched or unbatched)
    fx = sample["fixations"]
    xy_px: torch.Tensor = fx["xy"]
    xy_norm: torch.Tensor | None = fx.get("xy_norm")
    times: torch.Tensor = fx["time"]
    dwell: torch.Tensor = fx["dwell"]
    seg_hits: torch.Tensor = fx["seg_hits"]
    box_hits: torch.Tensor = fx["box_hits"]

    # If batched: (B, T, ...) or (B, T) -> take first sample
    if xy_px.dim() == 3:
        xy_px = xy_px[0]
    if isinstance(xy_norm, torch.Tensor) and xy_norm.dim() == 3:
        xy_norm = xy_norm[0]
    if times.dim() == 2:
        times = times[0]
    if dwell.dim() == 2:
        dwell = dwell[0]
    if isinstance(seg_hits, torch.Tensor) and seg_hits.dim() == 3:
        seg_hits = seg_hits[0]
    if isinstance(box_hits, torch.Tensor) and box_hits.dim() == 3:
        box_hits = box_hits[0]

    # Respect true sequence length if provided (to avoid padded zeros)
    true_len = None
    lengths = fx.get("lengths")
    if isinstance(lengths, torch.Tensor):
        if lengths.dim() == 0:
            true_len = int(lengths.item())
        elif lengths.dim() == 1 and lengths.numel() > 0:
            true_len = int(lengths[0].item())
    elif isinstance(lengths, (int, np.integer)):
        true_len = int(lengths)

    T = int(true_len if true_len is not None else xy_px.shape[0])
    T = max(0, min(T, xy_px.shape[0]))
    xy_px = xy_px[:T]
    times = times[:T]
    dwell = dwell[:T]
    if isinstance(xy_norm, torch.Tensor):
        xy_norm = xy_norm[:T]
    if seg_hits.ndim > 0:
        seg_hits = seg_hits[:T]
    if box_hits.ndim > 0:
        box_hits = box_hits[:T]

    # Now unbatched shapes: xy_px (T,2), times (T,), dwell (T,), seg_hits (T,S), box_hits (T,Cb)
    T = int(xy_px.shape[0])
    print(f"\nfixations: count={T}")

    if T > 0 and dwell.numel() > 0:
        d_np = dwell.detach().cpu().numpy().astype(np.float32)
        print(f"\ndwell (sec): mean={d_np.mean():.4f}, std={d_np.std():.4f}, min={d_np.min():.4f}, max={d_np.max():.4f}")

    # Print raw xy, image size, and normalized xy in [0,1]
    if T > 0:
        # infer H,W from image if available; else assume xy already in pixels scaled to the image size
        H = W = None
        img = sample.get("image")
        if isinstance(img, torch.Tensor) and img.dim() >= 3:
            H = int(img.shape[-2])
            W = int(img.shape[-1])
        # Print xy_raw stats
        if xy_px.dim() == 2 and xy_px.shape[1] >= 2:
            xy_raw = xy_px[:, :2].detach().cpu().tolist()
            # avoid huge print
            preview_n = min(T, 16)
            print(f"\nxy_raw (first {preview_n}/{T}): {xy_raw[:preview_n]}")
            x_vals = xy_px[:, 0].detach().cpu().numpy()
            y_vals = xy_px[:, 1].detach().cpu().numpy()
            print(f"\nxy_raw stats: x[min={x_vals.min():.1f}, max={x_vals.max():.1f}], y[min={y_vals.min():.1f}, max={y_vals.max():.1f}]")
    else:
            print("\nxy_raw: unavailable or wrong shape")
        print(f"\nimage tensor size (H,W): {H} x {W}")

        print(f"\ntimes (sec) [{T}]: {times.detach().cpu().tolist()}")

        if isinstance(xy_norm, torch.Tensor) and xy_norm.ndim == 2 and xy_norm.shape[1] >= 2:
            xy_norm_clamped = xy_norm.detach().float().clamp(0.0, 1.0)
            preview_n = min(T, 16)
            print(f"\nxy_norm dataset [0,1] (first {preview_n}/{T}): {xy_norm_clamped[:preview_n].cpu().tolist()}")

            if H is not None and W is not None and H > 0 and W > 0:
                scale_x = float(W - 1) if W and W > 1 else 1.0
                scale_y = float(H - 1) if H and H > 1 else 1.0
                x_px_tensor = (xy_norm_clamped[:, 0] * scale_x).cpu().tolist()
                y_px_tensor = (xy_norm_clamped[:, 1] * scale_y).cpu().tolist()
                preview_n = min(T, 16)
                xy_tensor_coords = [[x_px_tensor[i], y_px_tensor[i]] for i in range(preview_n)]
                print(f"\nxy on tensor grid (first {preview_n}/{T}): {xy_tensor_coords}")
        elif xy_px.dim() == 2 and xy_px.shape[1] >= 2 and H and W and H > 0 and W > 0:
            x_all = xy_px[:, 0].detach().float()
            y_all = xy_px[:, 1].detach().float()
            x_norm = (x_all / float(W - 1)).clamp(0.0, 1.0)
            y_norm = (y_all / float(H - 1)).clamp(0.0, 1.0)
            xy_norm_list = torch.stack([x_norm, y_norm], dim=1).cpu().tolist()
            print(f"\nxy_norm approximated [0,1] by tensor size [{T}]: {xy_norm_list[:min(T, 16)]}")

        # If a reference grid (segmentation or dicom) is available, also normalize by that
        ref_h = int(seg_h) if seg_h else (int(dicom_h) if dicom_h else None)
        ref_w = int(seg_w) if seg_w else (int(dicom_w) if dicom_w else None)
        if ref_h and ref_w and xy_px.dim() == 2 and xy_px.shape[1] >= 2:
            x_norm2 = (xy_px[:, 0].detach().float() / float(max(ref_w - 1, 1))).clamp(0.0, 1.0)
            y_norm2 = (xy_px[:, 1].detach().float() / float(max(ref_h - 1, 1))).clamp(0.0, 1.0)
            xy_norm2_list = torch.stack([x_norm2, y_norm2], dim=1).cpu().tolist()
            label = "segmentation grid" if seg_h and seg_w else "DICOM"
            print(f"\n{label} size (H,W): {ref_h} x {ref_w}")
            preview_n = min(T, 16)
            print(f"\nxy_norm_{label.replace(' ', '_')} [0,1] (first {preview_n}/{T}): {xy_norm2_list[:preview_n]}")

    if T > 0:
        # First fixation details (normalized if possible)
        if isinstance(xy_norm, torch.Tensor) and xy_norm.ndim == 2 and xy_norm.shape[1] >= 2:
            x0 = float(xy_norm[0, 0].clamp(0.0, 1.0).item())
            y0 = float(xy_norm[0, 1].clamp(0.0, 1.0).item())
        elif 'H' in locals() and H and W and xy_px.shape[1] >= 2:
            x0 = float((xy_px[0, 0] / float(max(W - 1, 1))).clamp(0.0, 1.0).item())
            y0 = float((xy_px[0, 1] / float(max(H - 1, 1))).clamp(0.0, 1.0).item())
        else:
            x0 = float(xy_px[0, 0].item()) if xy_px.shape[1] > 0 else float("nan")
            y0 = float(xy_px[0, 1].item()) if xy_px.shape[1] > 1 else float("nan")
        t0 = float(times[0].item()) if times.numel() > 0 else float("nan")
        d0 = float(dwell[0].item()) if dwell.numel() > 0 else float("nan")
        print(f"\nfirst fixation: xy_norm=({x0:.3f},{y0:.3f}) time={t0:.4f}s dwell={d0:.4f}s")

        if isinstance(seg_hits, torch.Tensor) and seg_hits.numel() > 0:
            print(f"\nseg_hits shape: {tuple(seg_hits.shape)} | first: {seg_hits[0].int().tolist()}")
            seg_frac = seg_hits.detach().float().mean(dim=0).cpu().tolist()
            print(f"\nseg_hits fraction per segment (mean over T): {seg_frac}")
        else:
            print("no seg_hits available")

        if isinstance(box_hits, torch.Tensor) and box_hits.numel() > 0:
            print(f"\nbox_hits shape: {tuple(box_hits.shape)} | first: {box_hits[0].int().tolist()}")
            box_frac = box_hits.detach().float().mean(dim=0).cpu().tolist()
            print(f"\nbox_hits fraction per class (mean over T): {box_frac}")
        else:
            print("no box_hits available")

    # Transcript
    tr = sample.get("transcript", {})
    print(f"\ntranscript-raw: {tr}")
    text = tr.get("text", "") if isinstance(tr, dict) else ""
    segs = tr.get("segments", []) if isinstance(tr, dict) else []
    print(f"\ntranscript: chars={len(text)} | segments={len(segs)}")
    for i, seg in enumerate(segs[:5]):
        begin = seg.get("begin")
        end = seg.get("end")
        content = seg.get("text", "")
        print(f"  seg[{i}]: begin={begin} end={end} text='{content[:80]}'")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--max-fixations", type=int, default=64)
    ap.add_argument("--classes", nargs="+", default=["CHF", "pneumonia", "Normal"])
    ap.add_argument("--save-json", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "sample")
    args = ap.parse_args()

    cfg = ConfigLoader(args.config)

    # Splits
    split_dir = Path(cfg.get("split_files", "dir", default=ROOT / "configs" / "splits"))
    if not split_dir.is_absolute():
        split_dir = ROOT / split_dir
    train_ids = read_split_ids(split_dir, "train")
    val_ids = read_split_ids(split_dir, "val")
    test_ids = read_split_ids(split_dir, "test")

    # Build datasets
    root = Path(cfg.get("input_path", "gaze_raw"))
    seg = Path(cfg.get("input_path", "segmentation_dir"))
    transcripts = Path(cfg.get("input_path", "transcripts_dir", default=seg))
    dicom_root = Path(cfg.get("input_path", "dicom_raw"))

    def build(case_ids):
        return EGDCXRDataset(
            root=root,
            seg_path=seg,
            transcripts_path=transcripts,
            dicom_root=dicom_root,
            max_fixations=args.max_fixations,
            case_ids=case_ids,
            classes=args.classes,
        )

    print("Loading datasets...")
    dtr = build(train_ids)
    dval = build(val_ids)
    dtest = build(test_ids)
    print(f"Totals | train={len(dtr)} | val={len(dval)} | test={len(dtest)} | classes={args.classes}")

    # Create a small loader and take one batch
    ltr = create_dataloader(dtr, batch_size=args.batch_size, shuffle=False, sampler=None, num_workers=args.num_workers)
    batch = next(iter(ltr))

    # Take first sample from batch and normalize keys to match __getitem__ layout for printing
    bs = batch["labels"]["single_index"].shape[0]
    sample: Dict[str, Any] = {k: (v[0] if isinstance(v, torch.Tensor) and v.shape[0] == bs else v) for k, v in batch.items()}
    # Align keys similar to __getitem__ layout
    if "images" in sample:
        sample["image"] = sample.pop("images")
    if "dicom_ids" in sample:
        sample["dicom_id"] = sample.pop("dicom_ids")[0]
                    if "transcripts" in sample:
        sample["transcript"] = sample.pop("transcripts")[0]
    # Bring metadata if provided by dataset/batch
    if "meta" in batch:
        meta_src = batch["meta"]
        if isinstance(meta_src, dict):
            meta_out = {}
            for k, v in meta_src.items():
                if isinstance(v, torch.Tensor):
                    meta_out[k] = v[0].item() if v.dim() == 1 and v.shape[0] == bs else (
                        v.item() if v.dim() == 0 else v[0].item() if v.shape[0] > 0 else None
                    )
                else:
                    meta_out[k] = v
            sample["meta"] = meta_out
        else:
            sample["meta"] = meta_src
    if "fixations" in sample and isinstance(sample["fixations"], dict):
        fx_dict = sample["fixations"]
        sample["fixations"] = {
            key: (val[0] if isinstance(val, torch.Tensor) and val.shape[0] == bs else val)
            for key, val in fx_dict.items()
        }
        lengths_val = sample["fixations"].get("lengths")
        if isinstance(lengths_val, torch.Tensor):
            if lengths_val.numel() == 1:
                sample["fixations"]["lengths"] = int(lengths_val.item())
            else:
                sample["fixations"]["lengths"] = int(lengths_val[0].item())

    inspect_example(sample)

    # Save to sample/ as JSON if requested (always save one example as per request)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sample_000.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_make_json_serializable(sample), f, indent=2)
    print(f"\n💾 Saved inspected example to: {out_path}")


if __name__ == "__main__":
    main()
