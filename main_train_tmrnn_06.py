#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys, yaml, os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

# local imports
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from egd_cxr_dataset.datasets.egd_cxr import EGDCXRDataset, create_dataloader
from src.models.tmrnn import TMRNN, build_vocabulary


def set_seed(seed: int = 2025):
    import random, numpy as np
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def read_split_ids(split_dir: Path, split: str) -> List[str]:
    path = split_dir / f"{split}_ids.txt"
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Split file not found: {path}")
    ids: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if s and not s.startswith("#"):
            ids.append(s)
    if not ids:
        raise FileNotFoundError(f"[ERROR] Split file exists but is empty: {path}")
    return ids

def encode_batch_transcripts(transcripts, tokenizer, max_len: int, device: torch.device) -> torch.Tensor:
    """Match gold: pad-only when empty, otherwise standard encode."""
    ids = []
    pad = tokenizer.pad_token_id
    for tr in transcripts:
        txt = (tr.get("text") or "").strip()
        if not txt:
            ids.append(torch.full((max_len,), pad, dtype=torch.long))
        else:
            enc = tokenizer(
                txt, max_length=max_len, truncation=True,
                padding="max_length", return_tensors="pt"
            )
            ids.append(enc.input_ids.squeeze(0))
    return torch.stack(ids).to(device)

def run_one_epoch(
    model: TMRNN,
    loader,
    tokenizer,
    cfg: Dict[str, Any],
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    train: bool = True,
) -> Tuple[float, float, float, float]:
    """
    Returns:
      total_loss, cls_loss, txt_loss, top1_acc
    """
    model.train(train)
    # Train uses label smoothing; Val uses plain CE (gold behavior)
    ce_cls = nn.CrossEntropyLoss(label_smoothing=float(cfg["train"].get("label_smoothing", 0.0))) if train else nn.CrossEntropyLoss()
    total_loss = total_cls = total_txt = 0.0
    total_correct = 0
    total_seen = 0

    torch.set_grad_enabled(train)
    for batch in tqdm(loader, disable=not train, leave=False):
        # move tensors
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)
            elif isinstance(v, dict):
                for sk, sv in v.items():
                    if isinstance(sv, torch.Tensor):
                        batch[k][sk] = sv.to(device, non_blocking=True)

        transcripts_ids = None
        if train and model.enable_text_head and model.use_teacher_forcing and tokenizer is not None:
            transcripts_ids = encode_batch_transcripts(
                batch["transcripts"], model.tok, cfg["train"]["max_txt_len"], device
            )

        cls_logits, txt_logits = model(batch, transcripts_ids=transcripts_ids)
        y = batch["labels"]["single_index"]

        # losses
        loss_cls = ce_cls(cls_logits, y)
        loss_txt = torch.tensor(0.0, device=device)
        # IMPORTANT: with labels=... inside T5, logits align to full labels (no [:,1:])
        if train and txt_logits is not None and transcripts_ids is not None:
            pad_id = model.tok.pad_token_id
            loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)
            loss_txt = loss_fct(
                txt_logits.reshape(-1, txt_logits.size(-1)),
                transcripts_ids.reshape(-1)
            )
        loss = loss_cls + float(cfg["train"]["lambda_txt"]) * loss_txt

        # step
        if train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # accumulators
        total_loss += loss.item()
        total_cls  += loss_cls.item()
        total_txt  += float(loss_txt)

        with torch.no_grad():
            preds = cls_logits.argmax(dim=1)
            total_correct += (preds == y).float().sum().item()
            total_seen    += y.numel()

    denom = max(1, len(loader))
    acc = total_correct / max(1, total_seen)
    return total_loss / denom, total_cls / denom, total_txt / denom, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=2025)

    # Modality/backbone switches
    ap.add_argument("--image_backbone", type=str, default=None,
                    choices=["resnet18", "resnet50", "densenet121", "txrv_densenet121"])
    ap.add_argument("--no_gaze", action="store_true")
    ap.add_argument("--no_roi", action="store_true")
    ap.add_argument("--no_text", action="store_true", help="disable teacher-forcing")
    ap.add_argument("--no_text_decode", action="store_true", help="disable the text head entirely")

    args = ap.parse_args()
    set_seed(args.seed)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve split dir robustly (point this to the SAME splits as gold for apples-to-apples)
    project_root = Path(__file__).resolve().parent
    config_dir = args.config.parent.resolve()
    split_dir_raw = Path(cfg["split_files"]["dir"])
    if split_dir_raw.is_absolute():
        split_dir = split_dir_raw
    else:
        first = split_dir_raw.parts[0] if split_dir_raw.parts else ""
        if first.startswith("configs"):
            split_dir = (project_root / split_dir_raw).resolve()
        else:
            split_dir = (config_dir / split_dir_raw).resolve()

    train_ids = read_split_ids(split_dir, "train")
    val_ids   = read_split_ids(split_dir, "val")

    # Datasets / loaders
    in_paths = cfg["input_path"]
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

    bs = cfg["train"]["batch_size"]; nw = cfg["train"]["num_workers"]
    sampler = None
    if cfg["train"].get("use_weighted_sampler", False):
        w = train_ds.sample_weights()
        sampler = WeightedRandomSampler(w, len(w), replacement=True)

    train_loader = create_dataloader(train_ds, batch_size=bs, shuffle=(sampler is None), sampler=sampler, num_workers=nw)
    val_loader   = create_dataloader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw)

    # Build model with CLI overrides (CLI > YAML options > default)
    opts = cfg.get("options", {})
    def opt(key, default):
        v_cli = getattr(args, key, None)
        return v_cli if v_cli is not None else opts.get(key, default)

    model = TMRNN(
        num_seg=getattr(train_ds, "num_segments", None) or len(train_ds.region_names),
        num_box=train_ds.num_box_classes,
        d_g=cfg["model"]["d_g"],
        d_r=cfg["model"]["d_s"],
        d_img=cfg["model"]["d_img"],
        d_h=cfg["model"]["d_h"],
        num_classes=len(cfg["train"]["classes"]),
        image_backbone=opt("image_backbone", "resnet50"),
        use_gaze=not args.no_gaze if hasattr(args, "no_gaze") else True,
        use_roi=not args.no_roi if hasattr(args, "no_roi") else True,
        enable_text_head=not args.no_text_decode if hasattr(args, "no_text_decode") else True,
        use_teacher_forcing=not args.no_text if hasattr(args, "no_text") else True,
    ).to(device)

    # Vocab only needed if text head is enabled (kept for completeness)
    vocab = None
    if model.enable_text_head:
        vocab = build_vocabulary(train_ds, min_freq=2)

    optm = AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    # Train
    epochs = cfg["train"]["epochs"]
    patience = cfg["train"].get("patience", 8)
    best_val = float("inf"); bad = 0

    ckpt_dir = Path(cfg["output_path"]["checkpoint_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / cfg["output_path"].get("best_ckpt_name", "tmrnn_best.pt")
    if vocab is not None:
        (ckpt_dir / "meta").mkdir(parents=True, exist_ok=True)
        with (ckpt_dir / "meta" / "vocab.json").open("w") as f:
            json.dump(dict(word2idx=vocab.word2idx), f, indent=2)

    for ep in range(1, epochs + 1):
        tr_loss, tr_cls, tr_txt, tr_acc = run_one_epoch(
            model, train_loader, model.tok if model.enable_text_head else None,
            cfg, device, optimizer=optm, train=True
        )
        va_loss, va_cls, va_txt, va_acc = run_one_epoch(
            model, val_loader, None, cfg, device, optimizer=None, train=False
        )

        print(f"[Epoch {ep:02d}] "
              f"train loss={tr_loss:.4f} (cls={tr_cls:.4f}, txt={tr_txt:.4f}) acc={tr_acc*100:.2f}% | "
              f"val loss={va_loss:.4f} (cls={va_cls:.4f}, txt={va_txt:.4f}) acc={va_acc*100:.2f}%")

        if va_loss < best_val:
            best_val = va_loss; bad = 0
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": ep}, best_path)
            print(f"  → Saved best to {best_path}")
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping (patience={patience})."); break

    print("Training finished.")


if __name__ == "__main__":
    main()
