#!/usr/bin/env python3
"""
Weighted multi-label training with OneCycleLR scheduling and threshold tuning.

This script builds on main_train_st2.py by adding a OneCycleLR schedule for the
AdamW optimiser while retaining positive class weighting, dropout, and threshold tuning.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from egd_cxr_dataset import ConfigLoader, EGDCXRDataset, build_vocab, create_dataloader  # noqa: E402
from egd_cxr_dataset.models.gaze_intent_seq_rnn import (  # noqa: E402
    GazeSeqRNNAttend as GazeIntent2TranscriptAndLabels,
)


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_split_ids(split_dir: Path, split: str) -> List[str]:
    path = split_dir / f"{split}_ids.txt"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def gather_transcripts(dataset: EGDCXRDataset, case_ids: Iterable[str]) -> List[str]:
    texts: List[str] = []
    for case_id in case_ids:
        payload = dataset._load_transcript(case_id)  # pylint: disable=protected-access
        texts.append(payload.get("text", ""))
    return texts


def compute_pos_weight(dataset: EGDCXRDataset) -> torch.Tensor:
    pos: Optional[torch.Tensor] = None
    for idx in range(len(dataset)):
        y = dataset[idx]["labels"]["binary"].float()
        pos = y if pos is None else (pos + y)
    total = torch.tensor(len(dataset), dtype=torch.float32)
    neg = total - pos
    pos_weight = (neg / pos.clamp_min(1.0)).float()
    return pos_weight


def compute_class_accuracy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bin_pred = (pred >= 0.5).float()
    bin_target = (target >= 0.5).float()
    correct = (bin_pred == bin_target).float()
    return correct.mean(dim=0)


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.sum() == 0:
        return 0.0
    order = np.argsort(-y_score)
    y_true = y_true[order]
    tp = np.cumsum(y_true)
    fp = np.cumsum(1.0 - y_true)
    denom = np.maximum(tp + fp, 1e-6)
    precision = tp / denom
    recall = tp / (y_true.sum() + 1e-6)
    precision = np.concatenate(([precision[0]], precision))
    recall = np.concatenate(([0.0], recall))
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def compute_pr_auc(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    preds = pred.detach().cpu().numpy()
    targets = target.detach().cpu().numpy()
    macro_vals: List[float] = []
    for c in range(preds.shape[1]):
        ap = _average_precision(targets[:, c], preds[:, c])
        if ap > 0.0:
            macro_vals.append(ap)
    macro_pr = float(np.mean(macro_vals)) if macro_vals else 0.0
    micro_pr = _average_precision(targets.reshape(-1), preds.reshape(-1))
    return macro_pr, micro_pr


def compute_multilabel_summary(
    pred: torch.Tensor,
    target: torch.Tensor,
    thresholds: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    eps = 1e-6
    if thresholds is not None:
        thresh = thresholds.to(pred.device).view(1, -1)
        bin_pred = (pred >= thresh).float()
    else:
        bin_pred = (pred >= 0.5).float()
    bin_target = (target >= 0.5).float()

    tp = (bin_pred * bin_target).sum(dim=0)
    fp = (bin_pred * (1.0 - bin_target)).sum(dim=0)
    fn = ((1.0 - bin_pred) * bin_target).sum(dim=0)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    macro_acc = float(compute_class_accuracy(pred, target).mean().item())
    macro_f1 = float(f1.mean().item())

    tp_micro = float(tp.sum().item())
    fp_micro = float(fp.sum().item())
    fn_micro = float(fn.sum().item())

    micro_precision = tp_micro / (tp_micro + fp_micro + eps)
    micro_recall = tp_micro / (tp_micro + fn_micro + eps)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + eps)

    macro_pr_auc, micro_pr_auc = compute_pr_auc(pred, target)

    return {
        "macro_acc": macro_acc,
        "macro_f1": float(macro_f1),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "macro_pr_auc": float(macro_pr_auc),
        "micro_pr_auc": float(micro_pr_auc),
    }


def format_accuracy(label_names: List[str], acc: torch.Tensor) -> str:
    pieces = [f"{name}: {float(a):.3f}" for name, a in zip(label_names, acc.tolist())]
    return ", ".join(pieces)


def format_summary(summary: Dict[str, float]) -> str:
    return (
        f"macro-acc {summary['macro_acc']:.3f} | "
        f"macro-F1 {summary['macro_f1']:.3f} | "
        f"micro-P {summary['micro_precision']:.3f} | "
        f"micro-R {summary['micro_recall']:.3f} | "
        f"micro-F1 {summary['micro_f1']:.3f} | "
        f"macro-PR {summary['macro_pr_auc']:.3f} | "
        f"micro-PR {summary['micro_pr_auc']:.3f}"
    )


def tune_thresholds(
    probs: np.ndarray,
    targets: np.ndarray,
    grid: Optional[np.ndarray] = None,
) -> np.ndarray:
    grid = grid or np.linspace(0.05, 0.95, 19)
    C = probs.shape[1]
    best = np.full(C, 0.5, dtype=np.float32)
    for c in range(C):
        pc = probs[:, c]
        tc = targets[:, c]
        best_f1, best_t = 0.0, 0.5
        for t in grid:
            pred = (pc >= t).astype(np.float32)
            tp = (pred * tc).sum()
            fp = (pred * (1.0 - tc)).sum()
            fn = ((1.0 - pred) * tc).sum()
            f1 = (2 * tp) / (2 * tp + fp + fn + 1e-6)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best[c] = best_t
    return best


def run_epoch(
    model: GazeIntent2TranscriptAndLabels,
    loader,
    *,
    device: torch.device,
    vocab,
    optimiser: Optional[AdamW],
    desc: str = "",
    pos_weight: Optional[torch.Tensor] = None,
    thresholds: Optional[torch.Tensor] = None,
    scheduler: Optional[OneCycleLR] = None,
) -> Tuple[float, torch.Tensor, int, Dict[str, float], torch.Tensor, torch.Tensor]:
    train_mode = optimiser is not None
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_cases = 0
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    pos_weight_device = pos_weight.to(device) if pos_weight is not None else None

    progress = tqdm(loader, desc=desc or "epoch", leave=False)
    for batch in progress:
        fix = batch["fixations"]
        labels_batch = batch["labels"]["binary"].to(device)
        images = batch["images"].to(device)
        transcripts = batch["transcripts"]

        batch_loss = 0.0
        case_losses: List[torch.Tensor] = []
        preds_case: List[torch.Tensor] = []
        targets_case: List[torch.Tensor] = []

        for idx in range(labels_batch.size(0)):
            length = int(fix["lengths"][idx].item())
            if length == 0:
                continue

            xy = fix["xy"][idx, :length].to(device)
            dwell = fix["dwell"][idx, :length].to(device)
            time_s = fix["time"][idx, :length].to(device)
            seg_hits = fix["seg_hits"][idx, :length].to(device)
            box_hits = fix["box_hits"][idx, :length].to(device)

            image = images[idx]
            transcript = transcripts[idx]
            labels = labels_batch[idx]

            outputs = model.forward_case(
                fixations={
                    "xy": xy,
                    "dwell": dwell,
                    "time": time_s,
                    "seg_hits": seg_hits,
                    "box_hits": box_hits,
                },
                transcript=transcript,
                encode_text_fn=lambda s: vocab.encode(s, add_eos=True),
                image_1chw=image,
            )

            loss_labels = F.binary_cross_entropy_with_logits(
                outputs["label_logits"],
                labels.float(),
                pos_weight=pos_weight_device,
            )

            loss_text = torch.tensor(0.0, device=device)
            tok_total = 0
            if model.use_text:
                segments_iterable = transcript.get("segments", [])
                for logits, seg in zip(outputs["txt_logits_per_segment"], segments_iterable):
                    if logits.numel() == 0:
                        continue
                    tgt = vocab.encode(seg.get("text", ""), add_eos=True).to(device)
                    if tgt.numel() < 2:
                        continue
                    loss_text = loss_text + F.cross_entropy(logits, tgt[1:], reduction="sum")
                    tok_total += tgt.numel() - 1
                if tok_total > 0:
                    loss_text = loss_text / tok_total

            loss_speak = torch.tensor(0.0, device=device)
            if model.use_text and isinstance(transcript, dict) and transcript.get("segments"):
                speak_tgt = torch.zeros_like(outputs["speak_logits"])
                begins = [float(seg.get("begin", 0.0)) for seg in transcript.get("segments", [])]
                for b in begins:
                    nearest_idx = int(torch.argmin(torch.abs(time_s - b)).item())
                    speak_tgt[nearest_idx] = 1.0
                loss_speak = F.binary_cross_entropy_with_logits(outputs["speak_logits"], speak_tgt)

            loss = loss_labels + loss_text + 0.1 * loss_speak

            batch_loss += loss
            case_losses.append(loss)
            preds_case.append(torch.sigmoid(outputs["label_logits"]).detach().cpu())
            targets_case.append(labels.detach().cpu())
            total_cases += 1

        if not case_losses:
            continue

        loss_value = batch_loss / len(case_losses)

        if train_mode:
            optimiser.zero_grad(set_to_none=True)
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += float(loss_value.item()) * len(case_losses)
        all_preds.extend(preds_case)
        all_targets.extend(targets_case)

    if all_preds:
        pred_tensor = torch.stack(all_preds, dim=0)
        target_tensor = torch.stack(all_targets, dim=0)
    else:
        pred_tensor = torch.zeros(1, model.label_head.out_features)
        target_tensor = torch.zeros_like(pred_tensor)

    per_class_acc = compute_class_accuracy(pred_tensor, target_tensor)
    avg_loss = total_loss / max(1, total_cases)
    summary_metrics = compute_multilabel_summary(pred_tensor, target_tensor, thresholds=thresholds)
    return avg_loss, per_class_acc, total_cases, summary_metrics, pred_tensor, target_tensor


def build_model_and_vocab(
    train_dataset: EGDCXRDataset,
    device: torch.device,
    *,
    txt_dim: int,
    enc_dim: int,
    max_decode_len: int,
    use_bbox: bool,
    use_seg: bool,
    use_image: bool,
    use_text: bool,
    encoder_dropout: float,
    label_dropout: float,
):
    train_texts = gather_transcripts(train_dataset, train_dataset.case_ids)
    vocab = build_vocab(train_texts, min_freq=1, max_size=30000)

    sample = train_dataset[0]
    fix = sample["fixations"]
    num_segments = fix["seg_hits"].shape[1]
    num_box_classes = fix["box_hits"].shape[1]
    num_labels = sample["labels"]["binary"].shape[0]

    model = GazeIntent2TranscriptAndLabels(
        num_box_classes=num_box_classes,
        num_segments=num_segments,
        img_out_dim=enc_dim,
        intent_dim=enc_dim,
        vocab_size=vocab.size,
        dec_dim=txt_dim,
        num_labels=num_labels,
        pad_id=vocab.pad_id,
        bos_id=vocab.bos_id,
        eos_id=vocab.eos_id,
        use_box=use_bbox,
        use_seg=use_seg,
        use_image=use_image,
        use_text=use_text,
        encoder_dropout=encoder_dropout,
        label_dropout=label_dropout,
    ).to(device)

    return model, vocab


def save_checkpoint(
    path: Path,
    *,
    model: GazeIntent2TranscriptAndLabels,
    optimiser: AdamW,
    scheduler: Optional[OneCycleLR],
    vocab,
    label_names: List[str],
    epoch: int,
    config_path: Path,
    metrics: Dict[str, float],
    hparams: Dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vocab_state = {"itos": vocab.itos}
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimiser.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "vocab": vocab_state,
        "label_names": label_names,
        "config_path": str(config_path),
        "metrics": metrics,
        "hparams": hparams,
    }
    torch.save(checkpoint, path)
    print(f"  Checkpoint saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train silence-thought RNN with OneCycleLR on EGD-CXR data")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "data_egd-cxr.yaml", help="Path to configuration file")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Maximum learning rate for OneCycleLR")
    parser.add_argument("--weight-decay", type=float, default=3e-4, help="Weight decay")
    parser.add_argument("--max-fixations", type=int, default=64, help="Maximum fixations per case")
    parser.add_argument("--txt-dim", type=int, default=256, help="Text decoder dimension")
    parser.add_argument("--enc-dim", type=int, default=256, help="Encoder dimension")
    parser.add_argument("--max-decode-len", type=int, default=64, help="Maximum decode length")
    parser.add_argument("--num-workers", type=int, default=0, help="Data loading workers")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "runs" / "checkpoints", help="Checkpoint directory")
    parser.add_argument("--no-bbox", action="store_true", help="Disable bounding box features")
    parser.add_argument("--no-seg", action="store_true", help="Disable segmentation features")
    parser.add_argument("--no-image", action="store_true", help="Disable image features")
    parser.add_argument("--no-text", action="store_true", help="Disable transcript decoder")
    parser.add_argument("--encoder-dropout", type=float, default=0.1, help="Dropout on pooled encoder features")
    parser.add_argument("--label-dropout", type=float, default=0.2, help="Dropout before label head")
    parser.add_argument("--pct-start", type=float, default=0.1, help="OneCycleLR pct_start")
    args = parser.parse_args()

    args.use_bbox = not args.no_bbox
    args.use_seg = not args.no_seg
    args.use_image = not args.no_image
    args.use_text = not args.no_text

    config_loader = ConfigLoader(args.config)

    train_cfg = config_loader.get("train", default={})
    if isinstance(train_cfg, dict):
        override_fields = {
            "batch_size": train_cfg.get("batch_size"),
            "epochs": train_cfg.get("epochs"),
            "lr": train_cfg.get("lr"),
            "weight_decay": train_cfg.get("weight_decay"),
            "max_fixations": train_cfg.get("max_fixations"),
            "txt_dim": train_cfg.get("txt_dim"),
            "enc_dim": train_cfg.get("enc_dim"),
            "max_decode_len": train_cfg.get("max_decode_len"),
            "num_workers": train_cfg.get("num_workers"),
            "seed": train_cfg.get("seed"),
            "checkpoint_dir": train_cfg.get("checkpoint_dir"),
            "encoder_dropout": train_cfg.get("encoder_dropout"),
            "label_dropout": train_cfg.get("label_dropout"),
            "pct_start": train_cfg.get("pct_start"),
        }
        for field, value in override_fields.items():
            if value is not None:
                if field == "checkpoint_dir":
                    setattr(args, field, Path(value))
                else:
                    setattr(args, field, value)
        bool_overrides = {
            "use_bbox": train_cfg.get("use_bbox"),
            "use_seg": train_cfg.get("use_seg"),
            "use_image": train_cfg.get("use_image"),
            "use_text": train_cfg.get("use_text"),
        }
        for field, value in bool_overrides.items():
            if value is not None:
                setattr(args, field, bool(value))
    elif train_cfg is not None:
        raise TypeError("Expected 'train' section in config to be a mapping")

    args.checkpoint_dir = Path(args.checkpoint_dir)
    if not args.checkpoint_dir.is_absolute():
        args.checkpoint_dir = ROOT / args.checkpoint_dir

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    gaze_root = Path(config_loader.get("input_path", "gaze_raw"))
    seg_dir = Path(config_loader.get("input_path", "segmentation_dir"))
    transcripts_dir = Path(config_loader.get("input_path", "transcripts_dir", default=seg_dir))
    dicom_root = Path(config_loader.get("input_path", "dicom_raw"))

    split_dir_cfg = config_loader.get("split_files", "dir", default=ROOT / "config" / "splits")
    split_dir = Path(split_dir_cfg)
    if not split_dir.is_absolute():
        split_dir = ROOT / split_dir

    train_ids = read_split_ids(split_dir, "train")
    val_ids = read_split_ids(split_dir, "val")
    test_ids = read_split_ids(split_dir, "test")
    print(f"Data splits - Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    print("Loading training dataset...")
    train_dataset = EGDCXRDataset(
        root=gaze_root,
        seg_path=seg_dir,
        transcripts_path=transcripts_dir,
        dicom_root=dicom_root,
        max_fixations=args.max_fixations,
        case_ids=train_ids,
    )

    print("Loading validation dataset...")
    val_dataset = EGDCXRDataset(
        root=gaze_root,
        seg_path=seg_dir,
        transcripts_path=transcripts_dir,
        dicom_root=dicom_root,
        max_fixations=args.max_fixations,
        case_ids=val_ids,
    )

    print("Loading test dataset...")
    test_dataset = EGDCXRDataset(
        root=gaze_root,
        seg_path=seg_dir,
        transcripts_path=transcripts_dir,
        dicom_root=dicom_root,
        max_fixations=args.max_fixations,
        case_ids=test_ids,
    )

    print("Computing positive class weights...")
    pos_weight = compute_pos_weight(train_dataset)
    print(f"  pos_weight (first five): {pos_weight[:5].tolist()}")

    print("Creating data loaders...")
    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = create_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = create_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("Building model and vocabulary...")
    model, vocab = build_model_and_vocab(
        train_dataset,
        device,
        txt_dim=args.txt_dim,
        enc_dim=args.enc_dim,
        max_decode_len=args.max_decode_len,
        use_bbox=args.use_bbox,
        use_seg=args.use_seg,
        use_image=args.use_image,
        use_text=args.use_text,
        encoder_dropout=args.encoder_dropout,
        label_dropout=args.label_dropout,
    )
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Vocabulary size: {vocab.size}")

    optimiser = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimiser,
        max_lr=args.lr,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        pct_start=args.pct_start,
    )

    label_names = train_dataset.label_proc.schema.class_columns
    print(f"Training on {len(label_names)} disease classes: {label_names}")

    print(f"\nStarting training for {args.epochs} epochs...")
    checkpoint_dir = args.checkpoint_dir
    run_id = time.strftime("%Y%m%d-%H%M%S")
    best_val_loss = float("inf")
    best_checkpoint_path: Optional[Path] = None
    last_train_loss = float("nan")
    last_val_loss = float("nan")
    val_summary = {"macro_acc": 0.0, "macro_f1": 0.0, "micro_f1": 0.0, "micro_precision": 0.0, "micro_recall": 0.0, "macro_pr_auc": 0.0, "micro_pr_auc": 0.0}

    hparams = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_fixations": args.max_fixations,
        "txt_dim": args.txt_dim,
        "enc_dim": args.enc_dim,
        "max_decode_len": args.max_decode_len,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "use_bbox": args.use_bbox,
        "use_seg": args.use_seg,
        "use_image": args.use_image,
        "use_text": args.use_text,
        "encoder_dropout": args.encoder_dropout,
        "label_dropout": args.label_dropout,
        "pct_start": args.pct_start,
    }

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc, _, train_summary, *_ = run_epoch(
            model,
            train_loader,
            device=device,
            vocab=vocab,
            optimiser=optimiser,
            desc=f"train {epoch:02d}",
            pos_weight=pos_weight,
            scheduler=scheduler,
        )

        val_loss, val_acc, _, val_summary, *_ = run_epoch(
            model,
            val_loader,
            device=device,
            vocab=vocab,
            optimiser=None,
            desc=f"val {epoch:02d}",
            pos_weight=pos_weight,
        )

        epoch_duration = time.perf_counter() - epoch_start
        last_train_loss = train_loss
        last_val_loss = val_loss
        metrics = {
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "epoch_seconds": float(epoch_duration),
            "train_macro_acc": train_summary["macro_acc"],
            "train_macro_f1": train_summary["macro_f1"],
            "train_micro_f1": train_summary["micro_f1"],
            "train_micro_precision": train_summary["micro_precision"],
            "train_micro_recall": train_summary["micro_recall"],
            "val_macro_acc": val_summary["macro_acc"],
            "val_macro_f1": val_summary["macro_f1"],
            "val_micro_f1": val_summary["micro_f1"],
            "val_micro_precision": val_summary["micro_precision"],
            "val_micro_recall": val_summary["micro_recall"],
            "val_macro_pr_auc": val_summary["macro_pr_auc"],
            "val_micro_pr_auc": val_summary["micro_pr_auc"],
        }
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_checkpoint_path = checkpoint_dir / f"{run_id}_best.pt"
            save_checkpoint(
                best_checkpoint_path,
                model=model,
                optimiser=optimiser,
                scheduler=scheduler,
                vocab=vocab,
                label_names=label_names,
                epoch=epoch,
                config_path=args.config,
                metrics=metrics,
                hparams=hparams,
            )

        print(
            f"Epoch {epoch:02d} | train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} | time {epoch_duration:.2f}s"
        )
        print("  Train summary: " + format_summary(train_summary))
        print("  Val summary:   " + format_summary(val_summary))
        print("  Train accuracy per class:")
        print("    " + format_accuracy(label_names, train_acc))
        print("  Val accuracy per class:")
        print("    " + format_accuracy(label_names, val_acc))

    print("\nEvaluating on test set...")
    test_loss, test_acc, batches, test_summary, *_ = run_epoch(
        model,
        test_loader,
        device=device,
        vocab=vocab,
        optimiser=None,
        desc="test",
        pos_weight=pos_weight,
    )
    print(f"Test loss {test_loss:.4f} over {batches} batches")
    print("Test summary:   " + format_summary(test_summary))

    print("Test accuracy per class:")
    print("  " + format_accuracy(label_names, test_acc))

    final_checkpoint_path = checkpoint_dir / f"{run_id}_last.pt"
    final_metrics = {
        "train_loss": float(last_train_loss),
        "val_loss": float(last_val_loss),
        "test_loss": float(test_loss),
        "train_summary": train_summary,
        "val_summary": val_summary,
        "test_summary": test_summary,
    }
    save_checkpoint(
        final_checkpoint_path,
        model=model,
        optimiser=optimiser,
        scheduler=scheduler,
        vocab=vocab,
        label_names=label_names,
        epoch=args.epochs,
        config_path=args.config,
        metrics=final_metrics,
        hparams=hparams,
    )
    if best_checkpoint_path is not None and best_checkpoint_path != final_checkpoint_path:
        print(f"\nBest validation checkpoint: {best_checkpoint_path}")
    print(f"Final checkpoint: {final_checkpoint_path}")


if __name__ == "__main__":
    main()
