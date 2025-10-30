#!/usr/bin/env python3
"""
Training entry-point for TMRNN (EGD-CXR).
Now supports:
  --image_backbone [resnet18|resnet50|densenet121|txrv_densenet121]
  --no_gaze          → drop gaze embedding
  --no_roi           → drop ROI (seg+box) projection
  --no_text          → no teacher-forcing (no transcript input), but keep decoder head
  --no_text_decode   → classification-only (no decoder head at all)
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import sys

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
import yaml
import os

# ==== Ensure local src import works (fix for ModuleNotFoundError) =====
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# =====================================================================

# ------------------------------------------------------------------ #
# Dataset & model
# ------------------------------------------------------------------ #
from egd_cxr_dataset.datasets.egd_cxr import EGDCXRDataset, create_dataloader
import importlib
model_candidates = ["src.models.tmrnn"]
TMRNN_mod = None
for modname in model_candidates:
    try:
        TMRNN_mod = importlib.import_module(modname)
        TMRNN = getattr(TMRNN_mod, "TMRNN")
        break
    except Exception:
        continue
from src.models.tmrnn import build_vocabulary, encode_with_tau

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def set_seed(seed: int = 2025):
    import random, numpy as np
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def read_split_ids(split_dir: Path, split: str) -> List[str]:
    path = split_dir / f"{split}_ids.txt"
    print(f"[INFO] Looking for split: {split} at {path}")
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Split file not found: {path} (pwd={Path.cwd()})\n"
                              f"Please check your config's split_files.dir value and that the split names match train/val/test exactly.\n"
                              f"If running via sbatch or relative to YAML, ensure the split folder path is right.")
    ids: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if s and not s.startswith("#"): ids.append(s)
    if not ids:
        raise FileNotFoundError(f"[ERROR] Split file exists but is empty: {path}")
    return ids

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    correct = (pred == targets).float().sum().item()
    return correct / max(1, targets.numel())

# ------------------------------------------------------------------ #
# Encode a batch of transcripts (used only when teacher-forcing is on)
# ------------------------------------------------------------------ #
def encode_batch_transcripts(
    transcripts: List[Dict[str, Any]],
    tokenizer,
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    ids = []
    for tr in transcripts:
        txt = tr.get("text", "")
        enc = tokenizer(
            txt,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        ids.append(enc.input_ids.squeeze(0))
    return torch.stack(ids).to(device)

# ------------------------------------------------------------------ #
# One epoch (train / val)
# ------------------------------------------------------------------ #
def run_one_epoch(
    model: TMRNN,
    loader,
    tokenizer,
    cfg: Dict[str, Any],
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    train: bool = True,
) -> Tuple[float, float, float]:
    model.train(train)
    ce_cls = nn.CrossEntropyLoss(label_smoothing=float(cfg["train"].get("label_smoothing", 0.0)))
    total_loss = total_cls = total_txt = n = 0.0

    for batch in tqdm(loader, disable=not train, leave=False):
        # -------------------------------------------------------------- #
        # Move everything to device
        # -------------------------------------------------------------- #
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)
            elif isinstance(v, dict):
                for sk, sv in v.items():
                    if isinstance(sv, torch.Tensor):
                        batch[k][sk] = sv.to(device, non_blocking=True)

        # -------------------------------------------------------------- #
        # Teacher-forcing transcript input (only if both flags allow it)
        # -------------------------------------------------------------- #
        transcripts_ids = None
        if model.use_teacher_forcing and model.enable_text_head:
            transcripts_ids = encode_batch_transcripts(
                batch["transcripts"],
                tokenizer,
                cfg["train"]["max_txt_len"],
                device,
            )

        # -------------------------------------------------------------- #
        # Forward
        # -------------------------------------------------------------- #
        cls_logits, txt_logits = model(batch, transcripts_ids=transcripts_ids)

        y = batch["labels"]["single_index"]

        loss_cls = ce_cls(cls_logits, y)
        loss_txt = torch.tensor(0.0, device=device)

        if txt_logits is not None and transcripts_ids is not None:
            pad_id = tokenizer.pad_token_id
            loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)
            loss_txt = loss_fct(
                txt_logits.view(-1, txt_logits.size(-1)),
                transcripts_ids[:, 1:].reshape(-1),   # shift-right target
            )

        loss = loss_cls + float(cfg["train"]["lambda_txt"]) * loss_txt

        if train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        total_cls  += loss_cls.item()
        total_txt  += loss_txt.item()
        n += 1

    return (
        total_loss / max(1, n),
        total_cls  / max(1, n),
        total_txt  / max(1, n),
    )

# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2025)

    # ------------------- NEW CLI FLAGS ------------------- #
    parser.add_argument(
        "--image_backbone",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet50", "densenet121", "txrv_densenet121"],
        help="Image encoder backbone",
    )
    parser.add_argument("--no_gaze", action="store_true", help="Disable gaze embedding")
    parser.add_argument("--no_roi", action="store_true", help="Disable ROI (seg+box) projection")
    parser.add_argument(
        "--no_text",
        action="store_true",
        help="Do NOT feed transcript tokens during training (no teacher-forcing)",
    )
    parser.add_argument(
        "--no_text_decode",
        action="store_true",
        help="Completely disable the text decoder head (classification-only)",
    )
    # ----------------------------------------------------- #

    args = parser.parse_args()
    set_seed(args.seed)

    cfg: Dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    project_root = Path(__file__).resolve().parent
    config_dir = args.config.parent.resolve()
    print(f"[INFO] Using config file: {args.config}\n[INFO] Config directory: {config_dir}")
    in_paths = cfg["input_path"]
    split_dir_raw = Path(cfg["split_files"]["dir"])
    # If relative and starts with 'configs/', interpret as from project root (not config_dir)
    if not split_dir_raw.is_absolute() and str(split_dir_raw).startswith("configs/"):
        split_dir = (project_root / split_dir_raw).resolve()
    else:
        split_dir = (split_dir_raw if split_dir_raw.is_absolute() else (config_dir / split_dir_raw)).resolve()
    print(f"[INFO] Split dir resolved to: {split_dir}")

    train_ids = read_split_ids(split_dir, "train")
    val_ids   = read_split_ids(split_dir, "val")
    test_ids  = read_split_ids(split_dir, "test")

    # ------------------------------------------------------------------ #
    # Datasets
    # ------------------------------------------------------------------ #
    common_kwargs = dict(
        root=Path(in_paths["gaze_raw"]),
        seg_path=Path(in_paths["segmentation_dir"]),
        transcripts_path=Path(in_paths["transcripts_dir"]),
        dicom_root=Path(in_paths["dicom_raw"]),
        max_fixations=cfg["train"]["max_fixations"],
        classes=cfg["train"]["classes"],
        drop_unlabelled=True,
    )
    train_ds = EGDCXRDataset(**common_kwargs, case_ids=train_ids)
    val_ds   = EGDCXRDataset(**common_kwargs, case_ids=val_ids)

    # ------------------------------------------------------------------ #
    # Vocabulary (only needed if text head is enabled)
    # ------------------------------------------------------------------ #
    vocab = None
    if not args.no_text_decode:
        vocab = build_vocabulary(train_ds, min_freq=2)

    # ------------------------------------------------------------------ #
    # Dataloaders
    # ------------------------------------------------------------------ #
    bs = cfg["train"]["batch_size"]
    nw = cfg["train"]["num_workers"]
    sampler = None
    if cfg["train"].get("use_weighted_sampler", False):
        w = train_ds.sample_weights()
        sampler = WeightedRandomSampler(w, len(w), replacement=True)

    train_loader = create_dataloader(train_ds, batch_size=bs, shuffle=(sampler is None), sampler=sampler, num_workers=nw)
    val_loader   = create_dataloader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw)

    # ------------------------------------------------------------------ #
    # Model – pass CLI flags
    # ------------------------------------------------------------------ #
    num_seg = getattr(train_ds, "num_segments", None) or len(train_ds.region_names)
    num_box = train_ds.num_box_classes

    # -- get YAML-based options (modality ablation, backbone) --
    options = cfg["options"] if "options" in cfg else {}
    # CLI override
    model_kwargs = dict(
        num_seg=num_seg,
        num_box=num_box,
        d_g=cfg["model"]["d_g"],
        d_r=cfg["model"]["d_s"],
        d_img=cfg["model"]["d_img"],
        d_h=cfg["model"]["d_h"],
        num_classes=len(cfg["train"]["classes"]),
    )
    def getopt(k, default=None):
        # CLI has highest precedence
        if hasattr(args, k) and getattr(args, k) is not None: return getattr(args, k)
        return options.get(k, default)
    model_kwargs.update({
        "image_backbone": getopt("image_backbone", "resnet50"),
        "use_gaze": not getattr(args, "no_gaze", False),
        "use_roi": not getattr(args, "no_roi", False),
        "enable_text_head": not getattr(args, "no_text_decode", False),
        "use_teacher_forcing": not getattr(args, "no_text", False),
    })
    # Only pass allowed kwargs for target TMRNN
    import inspect
    allowed_args = inspect.signature(TMRNN.__init__).parameters.keys()
    model_kwargs_filtered = {k: v for k, v in model_kwargs.items() if k in allowed_args}
    model = TMRNN(**model_kwargs_filtered)

    # ------------------------------------------------------------------ #
    # Optimizer & losses
    # ------------------------------------------------------------------ #
    opt = AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    # ------------------------------------------------------------------ #
    # Training loop with early stopping
    # ------------------------------------------------------------------ #
    epochs = cfg["train"]["epochs"]
    patience = cfg["train"].get("patience", 8)
    best_val = float("inf")
    bad = 0
    ckpt_dir = Path(cfg["output_path"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / (cfg["output_path"].get("best_ckpt_name") or "tmrnn_best.pt")

    # Save vocab (if any)
    if vocab is not None:
        (ckpt_dir / "meta").mkdir(parents=True, exist_ok=True)
        with (ckpt_dir / "meta" / "vocab.json").open("w") as f:
            json.dump(dict(word2idx=vocab.word2idx), f, indent=2)

    for ep in range(1, epochs + 1):
        tr_loss, tr_cls, tr_txt = run_one_epoch(
            model, train_loader, model.tok if hasattr(model, "tok") else None,
            cfg, model.device, optimizer, train=True
        )
        va_loss, va_cls, va_txt = run_one_epoch(
            model, val_loader, None, cfg, model.device, train=False
        )

        print(f"[Epoch {ep:02d}] "
              f"train loss={tr_loss:.4f} (cls={tr_cls:.4f}, txt={tr_txt:.4f}) | "
              f"val   loss={va_loss:.4f} (cls={va_cls:.4f}, txt={va_txt:.4f})")

        if va_loss < best_val:
            best_val = va_loss
            bad = 0
            torch.save({
                "model": model.state_dict(),
                "cfg": cfg,
                "epoch": ep,
                "val_loss": best_val,
                "vocab": getattr(vocab, "word2idx", None),
            }, best_path)
            print(f"  → Saved best checkpoint: {best_path}")
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping after {patience} epochs without improvement.")
                break

    print("Training finished.")


if __name__ == "__main__":
    main()