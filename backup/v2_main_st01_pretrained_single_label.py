#!/usr/bin/env python3
"""
Train ST01-style multimodal model for single-label classification on EGD-CXR.

This script mirrors the original ST01 training pipeline but reforms the task into
multi-class classification (default classes: CHF, Pneumonia, Normal). Metrics are
reported per class (accuracy with counts, precision, recall, F1) and at the macro /
micro aggregate level. AUC is computed using one-vs-rest (macro & micro) when
`sklearn` is available; otherwise those values are reported as NaN.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from egd_cxr_dataset import ConfigLoader, EGDCXRDataset, build_vocab  # type: ignore[import]
from egd_cxr_dataset.datasets.egd_cxr import collate_fn as multimodal_collate_fn  # type: ignore[import]
from egd_cxr_dataset.models.gaze_intent_seq_rnn import (  # type: ignore[import]
    GazeSeqRNNAttend as GazeIntent2TranscriptAndLabels,
)

from v2_src.single_label_dataset import (
    EGDCXRSingleLabelDataset,
    create_single_label_dataloader,
)

try:  # Optional dependency for AUC
    from sklearn.metrics import roc_auc_score

    HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    HAS_SKLEARN = False


# ----------------------------------------------------------------------------- #
# Utilities                                                                     #
# ----------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_metrics_header(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def compute_classification_report(
    logits: torch.Tensor,
    probs: torch.Tensor,
    targets: torch.Tensor,
    class_names: Sequence[str],
) -> Dict[str, torch.Tensor | float | None]:
    if logits.numel() == 0:
        empty = torch.zeros(len(class_names))
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
            "auc_macro": math.nan,
            "auc_micro": math.nan,
            "class_accuracy": empty,
            "class_precision": empty,
            "class_recall": empty,
            "class_f1": empty,
            "class_correct": empty,
            "class_total": empty,
        }

    num_classes = len(class_names)
    preds = probs.argmax(dim=1)

    # Confusion matrix
    cm = torch.zeros((num_classes, num_classes), dtype=torch.float64)
    for t, p in zip(targets.view(-1), preds.view(-1)):
        cm[int(t.item()), int(p.item())] += 1.0

    diag = cm.diag()
    actual = cm.sum(dim=1)
    predicted = cm.sum(dim=0)

    eps = 1e-8
    per_accuracy = torch.where(actual > 0, diag / (actual + eps), torch.zeros_like(diag))
    precision = torch.where(predicted > 0, diag / (predicted + eps), torch.zeros_like(diag))
    recall = torch.where(actual > 0, diag / (actual + eps), torch.zeros_like(diag))
    f1 = torch.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall + eps),
        torch.zeros_like(precision),
    )

    overall_accuracy = (diag.sum() / cm.sum().clamp_min(1.0)).item()
    macro_precision = precision.mean().item()
    macro_recall = recall.mean().item()
    macro_f1 = f1.mean().item()

    micro_precision = (diag.sum() / predicted.sum().clamp_min(1.0)).item()
    micro_recall = (diag.sum() / actual.sum().clamp_min(1.0)).item()
    micro_f1 = (
        2 * micro_precision * micro_recall / max(micro_precision + micro_recall, eps)
        if micro_precision + micro_recall > 0
        else 0.0
    )

    auc_macro: Optional[float] = None
    auc_micro: Optional[float] = None
    if HAS_SKLEARN:
        try:
            y_true = torch.nn.functional.one_hot(targets, num_classes=num_classes).cpu().numpy()
            auc_macro = float(roc_auc_score(y_true, probs.cpu().numpy(), average="macro", multi_class="ovr"))
            auc_micro = float(roc_auc_score(y_true, probs.cpu().numpy(), average="micro", multi_class="ovr"))
        except ValueError:
            auc_macro = math.nan
            auc_micro = math.nan
    else:
        auc_macro = math.nan
        auc_micro = math.nan

    return {
        "accuracy": overall_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "auc_macro": auc_macro,
        "auc_micro": auc_micro,
        "class_accuracy": per_accuracy,
        "class_precision": precision,
        "class_recall": recall,
        "class_f1": f1,
        "class_correct": diag,
        "class_total": actual,
    }


def print_class_report(
    report: Dict[str, torch.Tensor | float | None],
    class_names: Sequence[str],
    prefix: str,
) -> None:
    per_acc = report["class_accuracy"]
    per_prec = report["class_precision"]
    per_rec = report["class_recall"]
    per_f1 = report["class_f1"]
    per_correct = report["class_correct"]
    per_total = report["class_total"]

    assert isinstance(per_acc, torch.Tensor)
    assert isinstance(per_prec, torch.Tensor)
    assert isinstance(per_rec, torch.Tensor)
    assert isinstance(per_f1, torch.Tensor)
    assert isinstance(per_correct, torch.Tensor)
    assert isinstance(per_total, torch.Tensor)

    print(f"{prefix} per-class metrics:")
    for idx, class_name in enumerate(class_names):
        correct = int(per_correct[idx].item())
        total = int(per_total[idx].item())
        print(
            f"  {class_name}: acc {per_acc[idx]:.3f} ({correct}/{max(total,1)}) | "
            f"prec {per_prec[idx]:.3f} | rec {per_rec[idx]:.3f} | f1 {per_f1[idx]:.3f}"
        )

    print(
        f"{prefix} overall: acc {report['accuracy']:.3f} | "
        f"macro P/R/F1 {report['macro_precision']:.3f}/{report['macro_recall']:.3f}/{report['macro_f1']:.3f} | "
        f"micro P/R/F1 {report['micro_precision']:.3f}/{report['micro_recall']:.3f}/{report['micro_f1']:.3f} | "
        f"AUC macro/micro {report['auc_macro']:.3f}/{report['auc_micro']:.3f}"
    )


# ----------------------------------------------------------------------------- #
# Training epoch                                                                #
# ----------------------------------------------------------------------------- #
def run_epoch(
    model: GazeIntent2TranscriptAndLabels,
    loader,
    *,
    device: torch.device,
    vocab,
    optimiser: Optional[AdamW],
    class_weights: Optional[torch.Tensor],
    desc: str,
) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    train_mode = optimiser is not None
    model.train() if train_mode else model.eval()

    total_loss = 0.0
    total_cases = 0
    logits_list: List[torch.Tensor] = []
    targets_list: List[torch.Tensor] = []

    weight_vec = class_weights.to(device) if class_weights is not None else None

    progress = tqdm(loader, desc=desc, leave=False)
    for batch in progress:
        fix = batch["fixations"]
        label_indices = batch["labels"]["single_index"].to(device)
        images = batch["images"].to(device)
        transcripts = batch["transcripts"]

        case_losses: List[torch.Tensor] = []

        for idx in range(label_indices.size(0)):
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
            label = label_indices[idx]

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

            loss_label = F.cross_entropy(
                outputs["label_logits"].unsqueeze(0),
                label.view(1),
                weight=weight_vec,
            )

            # Text reconstruction loss (optional)
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

            # Speak gate auxiliary loss
            loss_speak = torch.tensor(0.0, device=device)
            if model.use_text and isinstance(transcript, dict) and transcript.get("segments"):
                speak_tgt = torch.zeros_like(outputs["speak_logits"])
                begins = [float(seg.get("begin", 0.0)) for seg in transcript.get("segments", [])]
                for begin in begins:
                    nearest_idx = int(torch.argmin(torch.abs(time_s - begin)).item())
                    speak_tgt[nearest_idx] = 1.0
                loss_speak = F.binary_cross_entropy_with_logits(outputs["speak_logits"], speak_tgt)

            total_case_loss = loss_label + loss_text + 0.1 * loss_speak
            case_losses.append(total_case_loss)

            logits_list.append(outputs["label_logits"].detach().cpu())
            targets_list.append(label.detach().cpu())
            total_cases += 1

        if not case_losses:
            continue

        batch_loss = torch.stack(case_losses).mean()
        if train_mode:
            optimiser.zero_grad(set_to_none=True)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

        total_loss += float(batch_loss.item()) * len(case_losses)

    if logits_list:
        logits_tensor = torch.stack(logits_list, dim=0)
        probs_tensor = torch.softmax(logits_tensor, dim=1)
        targets_tensor = torch.stack(targets_list, dim=0).long()
    else:
        logits_tensor = torch.zeros(0, model.label_head.out_features)
        probs_tensor = torch.zeros_like(logits_tensor)
        targets_tensor = torch.zeros(0, dtype=torch.long)

    avg_loss = total_loss / max(1, total_cases)
    return avg_loss, logits_tensor, probs_tensor, targets_tensor, total_cases


# ----------------------------------------------------------------------------- #
# Main                                                                          #
# ----------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-label ST01 training (image+bbox+seg+gaze+text)")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "data_egd-cxr.yaml",
        help="Path to experiment configuration file",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=2.5e-4)
    parser.add_argument("--max-fixations", type=int, default=64)
    parser.add_argument("--txt-dim", type=int, default=256)
    parser.add_argument("--enc-dim", type=int, default=256)
    parser.add_argument("--max-decode-len", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--classes",
        type=str,
        default="CHF,Pneumonia,Normal",
        help="Comma separated list of class names in priority order",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "runs" / "checkpoints_v2")
    parser.add_argument("--disable-class-weights", action="store_true")
    parser.add_argument("--disable-weighted-sampler", action="store_true")
    parser.add_argument("--pretrained-image", dest="pretrained_image", action="store_true")
    parser.add_argument("--no-pretrained-image", dest="pretrained_image", action="store_false")
    parser.set_defaults(pretrained_image=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_names = tuple(cls.strip() for cls in args.classes.split(",") if cls.strip())
    if not class_names:
        raise ValueError("At least one class must be provided via --classes.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config_loader = ConfigLoader(args.config)

    def cfg(*keys, default=None):
        return config_loader.get(*keys, default=default)

    gaze_root = Path(cfg("input_path", "gaze_raw", default="gaze_raw"))
    seg_dir = Path(cfg("input_path", "segmentation_dir", default="segmentation_dir"))
    transcripts_dir = Path(cfg("input_path", "transcripts_dir", default=seg_dir))
    dicom_root = Path(cfg("input_path", "dicom_raw", default="dicom_raw"))

    split_dir = Path(cfg("split_files", "dir", default=ROOT / "config" / "splits"))
    if not split_dir.is_absolute():
        split_dir = ROOT / split_dir

    def read_split(split: str) -> List[str]:
        path = split_dir / f"{split}_ids.txt"
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}")
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    train_ids = read_split("train")
    val_ids = read_split("val")
    test_ids = read_split("test")
    print(f"Data splits - Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    print("Loading datasets...")
    train_dataset = EGDCXRSingleLabelDataset(
        root=gaze_root,
        seg_path=seg_dir,
        transcripts_path=transcripts_dir,
        dicom_root=dicom_root,
        max_fixations=args.max_fixations,
        case_ids=train_ids,
        classes=class_names,
    )
    val_dataset = EGDCXRSingleLabelDataset(
        root=gaze_root,
        seg_path=seg_dir,
        transcripts_path=transcripts_dir,
        dicom_root=dicom_root,
        max_fixations=args.max_fixations,
        case_ids=val_ids,
        classes=class_names,
    )
    test_dataset = EGDCXRSingleLabelDataset(
        root=gaze_root,
        seg_path=seg_dir,
        transcripts_path=transcripts_dir,
        dicom_root=dicom_root,
        max_fixations=args.max_fixations,
        case_ids=test_ids,
        classes=class_names,
    )

    print("Creating data loaders...")
    class_weights = None if args.disable_class_weights else train_dataset.class_weights()

    sampler = None
    if not args.disable_weighted_sampler:
        weights = train_dataset.sample_weights().double()
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_dataset), replacement=True)

    train_loader = create_single_label_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
    )
    val_loader = create_single_label_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = create_single_label_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print("Building model and vocabulary...")
    train_texts = [train_dataset.base.label_proc.get_labels(cid).final_diagnosis or "" for cid in train_dataset.base.case_ids]
    vocab = build_vocab(train_texts, min_freq=1, max_size=30000)

    sample = train_dataset[0]
    fix = sample["fixations"]
    num_segments = fix["seg_hits"].shape[1]
    num_box_classes = fix["box_hits"].shape[1]
    num_classes = len(class_names)

    model = GazeIntent2TranscriptAndLabels(
        num_box_classes=num_box_classes,
        num_segments=num_segments,
        img_out_dim=args.enc_dim,
        intent_dim=args.enc_dim,
        vocab_size=vocab.size,
        dec_dim=args.txt_dim,
        num_labels=num_classes,
        pad_id=vocab.pad_id,
        bos_id=vocab.bos_id,
        eos_id=vocab.eos_id,
        use_box=True,
        use_seg=True,
        use_image=True,
        use_text=True,
        use_gaze=True,
        pretrained_image=args.pretrained_image,
        encoder_dropout=cfg("train", "encoder_dropout", default=0.1),
        label_dropout=cfg("train", "label_dropout", default=0.2),
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    optimiser = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("\nStarting training...")
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    best_val_loss = float("inf")
    best_checkpoint: Optional[Path] = None

    class_weight_tensor = class_weights if class_weights is None else class_weights.to(device)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_logits, train_probs, train_targets, _ = run_epoch(
            model,
            train_loader,
            device=device,
            vocab=vocab,
            optimiser=optimiser,
            class_weights=class_weight_tensor,
            desc=f"train {epoch:02d}",
        )

        val_loss, val_logits, val_probs, val_targets, _ = run_epoch(
            model,
            val_loader,
            device=device,
            vocab=vocab,
            optimiser=None,
            class_weights=class_weight_tensor,
            desc=f"val {epoch:02d}",
        )

        thresholds = None  # Per-class thresholds already applied within report via probabilities
        train_report = compute_classification_report(train_logits, train_probs, train_targets, class_names)
        val_report = compute_classification_report(val_logits, val_probs, val_targets, class_names)

        print(f"\nEpoch {epoch:02d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")
        print_class_report(train_report, class_names, prefix="Train")
        print_class_report(val_report, class_names, prefix="Val  ")

        is_best = val_loss < best_val_loss
        metrics_payload = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_report["accuracy"],
            "val_accuracy": val_report["accuracy"],
            "train_macro_f1": train_report["macro_f1"],
            "val_macro_f1": val_report["macro_f1"],
            "train_auc_macro": train_report["auc_macro"],
            "val_auc_macro": val_report["auc_macro"],
        }

        checkpoint_path = checkpoint_dir / f"{run_id}_epoch{epoch:02d}.pt"
        save_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimiser.state_dict(),
            "class_names": class_names,
            "config_path": str(args.config),
            "metrics": metrics_payload,
        }
        torch.save(save_payload, checkpoint_path)
        print(f"  Checkpoint saved to {checkpoint_path}")

        if is_best:
            best_val_loss = val_loss
            best_checkpoint = checkpoint_path
            print("  New best validation checkpoint.")

    format_metrics_header("Final evaluation on test set")
    _, test_logits, test_probs, test_targets, _ = run_epoch(
        model,
        test_loader,
        device=device,
        vocab=vocab,
        optimiser=None,
        class_weights=class_weight_tensor,
        desc="test",
    )
    test_report = compute_classification_report(test_logits, test_probs, test_targets, class_names)
    print_class_report(test_report, class_names, prefix="Test ")

    if best_checkpoint is not None:
        print(f"\nBest validation checkpoint: {best_checkpoint}")


if __name__ == "__main__":
    main()

