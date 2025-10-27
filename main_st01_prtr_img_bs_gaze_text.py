#!/usr/bin/env python3
"""
Train the silence-thought RNN on EGD-CXR with pretrained ResNet-18 image backbone for 3-class classification.

This script uses the rewritten dataset for CHF, pneumonia, and Normal classification.
It supports all modalities: gaze, bounding boxes, segmentation, transcripts, and images.
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
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from egd_cxr_dataset import ConfigLoader
from egd_cxr_dataset.utils.vocab import build_vocab
from src.datasets.egd_cxr_rewritten import EGDCXRRewrittenDataset, create_dataloader
from egd_cxr_dataset.models.gaze_intent_seq_rnn import (
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


def gather_transcripts(dataset: EGDCXRRewrittenDataset, case_ids: Iterable[str]) -> List[str]:
    texts: List[str] = []
    for case_id in case_ids:
        # Get transcript from dataset
        for idx in range(len(dataset)):
            master_idx = dataset._indices[idx]
            row = dataset.master_df.iloc[master_idx]
            if row["dicom_id"] == case_id:
                transcript = dataset._load_transcript(case_id)
                texts.append(transcript.get("text", ""))
                break
    return texts


def compute_classification_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute accuracy for 3-class classification."""
    pred_classes = torch.argmax(pred, dim=1)
    target_classes = target
    correct = (pred_classes == target_classes).float()
    return float(correct.mean().item())


def compute_classification_metrics(pred: torch.Tensor, target: torch.Tensor, class_names: List[str]) -> Dict[str, float]:
    """Compute detailed metrics for 3-class classification."""
    pred_classes = torch.argmax(pred, dim=1)
    target_classes = target
    
    # Sample counts
    total_samples = len(target_classes)
    correct_predictions = (pred_classes == target_classes).sum().item()
    
    # Overall accuracy
    accuracy = float(correct_predictions / total_samples)
    
    # Per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        class_mask = (target_classes == i)
        if class_mask.sum() > 0:
            class_pred = pred_classes[class_mask]
            class_target = target_classes[class_mask]
            class_acc = float((class_pred == class_target).float().mean().item())
            per_class_metrics[f"{class_name}_accuracy"] = class_acc
        else:
            per_class_metrics[f"{class_name}_accuracy"] = 0.0
    
    # Confusion matrix
    confusion_matrix = torch.zeros(len(class_names), len(class_names), dtype=torch.long)
    for i in range(len(pred_classes)):
        confusion_matrix[target_classes[i], pred_classes[i]] += 1
    
    # Precision, Recall, F1 for each class
    for i, class_name in enumerate(class_names):
        tp = confusion_matrix[i, i].item()
        fp = confusion_matrix[:, i].sum().item() - tp
        fn = confusion_matrix[i, :].sum().item() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class_metrics[f"{class_name}_precision"] = precision
        per_class_metrics[f"{class_name}_recall"] = recall
        per_class_metrics[f"{class_name}_f1"] = f1
    
    # Macro averages
    macro_precision = np.mean([per_class_metrics[f"{name}_precision"] for name in class_names])
    macro_recall = np.mean([per_class_metrics[f"{name}_recall"] for name in class_names])
    macro_f1 = np.mean([per_class_metrics[f"{name}_f1"] for name in class_names])
    
    # Compute AUC for each class (one-vs-rest)
    auc_scores = []
    for i, class_name in enumerate(class_names):
        # Create binary labels for this class
        binary_targets = (target_classes == i).float()
        binary_probs = pred[:, i]  # Probability for this class
        
        if binary_targets.sum() > 0 and (1 - binary_targets).sum() > 0:  # Both classes present
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(binary_targets.numpy(), binary_probs.numpy())
                auc_scores.append(auc)
                per_class_metrics[f"{class_name}_auc"] = auc
            except ImportError:
                per_class_metrics[f"{class_name}_auc"] = 0.0
        else:
            per_class_metrics[f"{class_name}_auc"] = 0.0
    
    macro_auc = np.mean(auc_scores) if auc_scores else 0.0
    
    return {
        "total_samples": total_samples,
        "correct_predictions": correct_predictions,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "macro_auc": macro_auc,
        **per_class_metrics
    }


def format_classification_metrics(metrics: Dict[str, float], class_names: List[str], split_name: str = "") -> str:
    """Format classification metrics for display."""
    total_samples = int(metrics["total_samples"])
    correct_predictions = int(metrics["correct_predictions"])
    
    lines = [
        f"{split_name.upper()} RESULTS:",
        f"  Samples: {total_samples} | Correct: {correct_predictions} | Accuracy: {metrics['accuracy']:.3f}",
        f"  Macro Precision: {metrics['macro_precision']:.3f} | Macro Recall: {metrics['macro_recall']:.3f} | Macro F1: {metrics['macro_f1']:.3f} | Macro AUC: {metrics['macro_auc']:.3f}",
        "",
        "Per-class metrics:"
    ]
    
    for class_name in class_names:
        acc = metrics[f"{class_name}_accuracy"]
        prec = metrics[f"{class_name}_precision"]
        rec = metrics[f"{class_name}_recall"]
        f1 = metrics[f"{class_name}_f1"]
        auc = metrics[f"{class_name}_auc"]
        lines.append(f"  {class_name:10s}: Acc={acc:.3f}, P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
    
    return "\n".join(lines)


def run_epoch(
    model: GazeIntent2TranscriptAndLabels,
    loader,
    *,
    device: torch.device,
    vocab,
    optimiser: Optional[AdamW],
    desc: str = "",
) -> Tuple[float, Dict[str, float], int]:
    train_mode = optimiser is not None
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_cases = 0
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    progress = tqdm(loader, desc=desc or "epoch", leave=False)
    for batch in progress:
        fix = batch["fixations"]
        # Use single_index for 3-class classification
        labels_batch = batch["labels"]["single_index"].to(device)
        images = batch["image"].to(device)
        transcripts = batch["transcript"]

        batch_loss = 0.0
        case_losses: List[torch.Tensor] = []
        preds_case: List[torch.Tensor] = []
        targets_case: List[torch.Tensor] = []

        for idx in range(labels_batch.size(0)):
            length = int(fix["lengths"][idx].item())
            if length == 0:
                continue

            try:
                xy = fix["xy"][idx, :length].to(device)
                dwell = fix["dwell"][idx, :length].to(device)
                time_s = fix["time"][idx, :length].to(device)
                seg_hits = fix["seg_hits"][idx, :length].to(device)
                box_hits = fix["box_hits"][idx, :length].to(device)

                image = images[idx]
                transcript = transcripts[idx]
                labels = labels_batch[idx]
            except Exception as e:
                print(f"Warning: Skipping sample {idx} in batch due to error: {e}")
                continue

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

            # Use cross-entropy loss for 3-class classification
            # Convert label_logits to classification logits (3 classes)
            classification_logits = outputs["label_logits"][:3]  # Take first 3 classes
            loss_labels = F.cross_entropy(classification_logits.unsqueeze(0), labels.unsqueeze(0))

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
            preds_case.append(F.softmax(classification_logits.unsqueeze(0), dim=1).detach().cpu())
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

        total_loss += float(loss_value.item()) * len(case_losses)
        all_preds.extend(preds_case)
        all_targets.extend(targets_case)

    if all_preds:
        pred_tensor = torch.cat(all_preds, dim=0)
        target_tensor = torch.stack(all_targets, dim=0)
    else:
        pred_tensor = torch.zeros(1, 3)  # 3 classes
        target_tensor = torch.zeros(1, dtype=torch.long)

    avg_loss = total_loss / max(1, total_cases)
    return avg_loss, pred_tensor, target_tensor, total_cases


def build_model_and_vocab(
    train_dataset: EGDCXRRewrittenDataset,
    device: torch.device,
    *,
    txt_dim: int,
    enc_dim: int,
    max_decode_len: int,
    use_bbox: bool,
    use_seg: bool,
    use_image: bool,
    use_text: bool,
    use_gaze: bool,
    pretrained_image: bool,
    encoder_dropout: float,
    label_dropout: float,
):
    # Get case IDs from dataset
    case_ids = []
    for idx in range(len(train_dataset)):
        master_idx = train_dataset._indices[idx]
        row = train_dataset.master_df.iloc[master_idx]
        case_ids.append(row["dicom_id"])
    
    train_texts = gather_transcripts(train_dataset, case_ids)
    vocab = build_vocab(train_texts, min_freq=1, max_size=30000)

    # Find a valid sample to get dimensions
    sample = None
    for idx in range(min(10, len(train_dataset))):  # Try first 10 samples
        try:
            sample = train_dataset[idx]
            break
        except (FileNotFoundError, Exception) as e:
            print(f"Warning: Skipping sample {idx} due to error: {e}")
            continue
    
    if sample is None:
        raise RuntimeError("Could not find any valid samples in the dataset")
    
    fix = sample["fixations"]
    num_segments = fix["seg_hits"].shape[1]
    num_box_classes = fix["box_hits"].shape[1]
    num_labels = 3  # CHF, pneumonia, Normal

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
        use_gaze=use_gaze,
        pretrained_image=pretrained_image,
        encoder_dropout=encoder_dropout,
        label_dropout=label_dropout,
    ).to(device)

    return model, vocab


def save_checkpoint(
    path: Path,
    *,
    model: GazeIntent2TranscriptAndLabels,
    optimiser: AdamW,
    vocab,
    class_names: List[str],
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
        "vocab": vocab_state,
        "class_names": class_names,
        "config_path": str(config_path),
        "metrics": metrics,
        "hparams": hparams,
    }
    torch.save(checkpoint, path)
    print(f"  Checkpoint saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train silence-thought RNN with pretrained image encoder for 3-class classification")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "st01_prtr_img_bs_gaze_text.yaml", help="Path to configuration file")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=2.5e-4, help="Weight decay for regularization")
    parser.add_argument("--max-fixations", type=int, default=64, help="Maximum number of fixations per case")
    parser.add_argument("--txt-dim", type=int, default=256, help="Text decoder dimension")
    parser.add_argument("--enc-dim", type=int, default=256, help="Encoder dimension")
    parser.add_argument("--max-decode-len", type=int, default=64, help="Maximum text generation length")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "runs" / "checkpoints", help="Directory to store checkpoints")
    parser.add_argument("--no-bbox", action="store_true", help="Disable bounding box features")
    parser.add_argument("--no-seg", action="store_true", help="Disable segmentation features")
    parser.add_argument("--no-image", action="store_true", help="Disable image features")
    parser.add_argument("--no-text", action="store_true", help="Disable transcript decoder")
    parser.add_argument("--no-gaze", action="store_true", help="Disable gaze kinematic features")
    parser.add_argument("--encoder-dropout", type=float, default=0.1, help="Dropout on pooled encoder features")
    parser.add_argument("--label-dropout", type=float, default=0.2, help="Dropout before label head")
    parser.add_argument("--pretrained-image", dest="pretrained_image", action="store_true", help="Use pretrained ResNet-18 image encoder")
    parser.add_argument("--no-pretrained-image", dest="pretrained_image", action="store_false", help="Disable pretrained image encoder")
    parser.set_defaults(pretrained_image=False)  # Default to False since DICOM images are not available
    args = parser.parse_args()

    args.use_bbox = not args.no_bbox
    args.use_seg = not args.no_seg
    args.use_image = not args.no_image
    args.use_text = not args.no_text
    args.use_gaze = not args.no_gaze

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
            "pretrained_image": train_cfg.get("pretrained_image"),
        }
        for field, value in override_fields.items():
            if value is not None:
                if field == "checkpoint_dir":
                    setattr(args, field, Path(value))
                elif field in ["lr", "weight_decay"]:
                    # Convert string scientific notation to float
                    setattr(args, field, float(value))
                else:
                    setattr(args, field, value)
        bool_overrides = {
            "use_bbox": train_cfg.get("use_bbox"),
            "use_seg": train_cfg.get("use_seg"),
            "use_image": train_cfg.get("use_image"),
            "use_text": train_cfg.get("use_text"),
            "use_gaze": train_cfg.get("use_gaze"),
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

    split_dir_cfg = config_loader.get("split_files", "dir", default=ROOT / "configs" / "splits")
    split_dir = Path(split_dir_cfg)
    if not split_dir.is_absolute():
        split_dir = ROOT / split_dir

    train_ids = read_split_ids(split_dir, "train")
    val_ids = read_split_ids(split_dir, "val")
    test_ids = read_split_ids(split_dir, "test")
    print(f"Data splits - Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    print("Loading training dataset...")
    train_dataset = EGDCXRRewrittenDataset(
        root=gaze_root,
        seg_path=seg_dir,
        transcripts_path=transcripts_dir,
        dicom_root=dicom_root,
        max_fixations=args.max_fixations,
        case_ids=train_ids,
        classes=["CHF", "pneumonia", "Normal"],
    )

    print("Loading validation dataset...")
    val_dataset = EGDCXRRewrittenDataset(
        root=gaze_root,
        seg_path=seg_dir,
        transcripts_path=transcripts_dir,
        dicom_root=dicom_root,
        max_fixations=args.max_fixations,
        case_ids=val_ids,
        classes=["CHF", "pneumonia", "Normal"],
    )

    print("Loading test dataset...")
    test_dataset = EGDCXRRewrittenDataset(
        root=gaze_root,
        seg_path=seg_dir,
        transcripts_path=transcripts_dir,
        dicom_root=dicom_root,
        max_fixations=args.max_fixations,
        case_ids=test_ids,
        classes=["CHF", "pneumonia", "Normal"],
    )

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
        use_gaze=args.use_gaze,
        pretrained_image=args.pretrained_image,
        encoder_dropout=args.encoder_dropout,
        label_dropout=args.label_dropout,
    )
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Vocabulary size: {vocab.size}")

    optimiser = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_names = ["CHF", "pneumonia", "Normal"]
    print(f"Training on {len(class_names)} classes: {class_names}")

    print(f"\nStarting training for {args.epochs} epochs...")
    checkpoint_dir = args.checkpoint_dir
    run_id = time.strftime("%Y%m%d-%H%M%S")
    best_val_loss = float("inf")
    best_checkpoint_path: Optional[Path] = None
    last_train_loss = float("nan")
    last_val_loss = float("nan")

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
        "use_gaze": args.use_gaze,
        "encoder_dropout": args.encoder_dropout,
        "label_dropout": args.label_dropout,
        "pretrained_image": args.pretrained_image,
    }

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_preds, train_targets, _ = run_epoch(
            model,
            train_loader,
            device=device,
            vocab=vocab,
            optimiser=optimiser,
            desc=f"train {epoch:02d}",
        )

        val_loss, val_preds, val_targets, _ = run_epoch(
            model,
            val_loader,
            device=device,
            vocab=vocab,
            optimiser=None,
            desc=f"val {epoch:02d}",
        )
        
        # Compute metrics
        train_metrics = compute_classification_metrics(train_preds, train_targets, class_names)
        val_metrics = compute_classification_metrics(val_preds, val_targets, class_names)
        
        epoch_duration = time.perf_counter() - epoch_start
        last_train_loss = train_loss
        last_val_loss = val_loss
        
        metrics = {
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "epoch_seconds": float(epoch_duration),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_checkpoint_path = checkpoint_dir / f"{run_id}_best.pt"
            save_checkpoint(
                best_checkpoint_path,
                model=model,
                optimiser=optimiser,
                vocab=vocab,
                class_names=class_names,
                epoch=epoch,
                config_path=args.config,
                metrics=metrics,
                hparams=hparams,
            )

        print(f"\nEpoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_duration:.2f}s")
        print("=" * 80)
        print(format_classification_metrics(train_metrics, class_names, "train"))
        print()
        print(format_classification_metrics(val_metrics, class_names, "val"))
        print("=" * 80)

    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION")
    print("=" * 80)
    test_loss, test_preds, test_targets, batches = run_epoch(
        model,
        test_loader,
        device=device,
        vocab=vocab,
        optimiser=None,
        desc="test",
    )
    
    test_metrics = compute_classification_metrics(test_preds, test_targets, class_names)
    print(f"Test Loss: {test_loss:.4f} | Batches: {batches}")
    print()
    print(format_classification_metrics(test_metrics, class_names, "test"))
    print("=" * 80)

    # Save test results to file
    test_results_file = checkpoint_dir / f"{run_id}_test_results.json"
    test_results = {
        "run_id": run_id,
        "config_path": str(args.config),
        "test_loss": float(test_loss),
        "test_metrics": test_metrics,
        "class_names": class_names,
        "hparams": hparams,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    import json
    with open(test_results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"\nTest results saved to: {test_results_file}")

    final_checkpoint_path = checkpoint_dir / f"{run_id}_last.pt"
    final_metrics = {
        "train_loss": float(last_train_loss),
        "val_loss": float(last_val_loss),
        "test_loss": float(test_loss),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    save_checkpoint(
        final_checkpoint_path,
        model=model,
        optimiser=optimiser,
        vocab=vocab,
        class_names=class_names,
        epoch=args.epochs,
        config_path=args.config,
        metrics=final_metrics,
        hparams=hparams,
    )
    
    print(f"\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    if best_checkpoint_path is not None and best_checkpoint_path != final_checkpoint_path:
        print(f"Best validation checkpoint: {best_checkpoint_path}")
    print(f"Final checkpoint: {final_checkpoint_path}")
    print(f"Test results: {test_results_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
