#!/usr/bin/env python3
"""
Train the silence-thought RNN on EGD-CXR with a pretrained ResNet-18 image backbone.
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

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from egd_cxr_dataset import ConfigLoader, EGDCXRDataset, build_vocab, create_dataloader
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


def gather_transcripts(dataset: EGDCXRDataset, case_ids: Iterable[str]) -> List[str]:
    texts: List[str] = []
    for case_id in case_ids:
        payload = dataset._load_transcript(case_id)  # pylint: disable=protected-access
        texts.append(payload.get("text", ""))
    return texts


def compute_class_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    thresholds: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if thresholds is not None:
        bin_pred = (pred >= thresholds.to(pred.device)).float()
    else:
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
        bin_pred = (pred >= thresholds.to(pred.device)).float()
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


def asymmetric_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma_pos: float = 0.0,
    gamma_neg: float = 4.0,
    clip: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Asymmetric focal loss for multilabel classification."""

    probs = torch.sigmoid(logits)
    pos_prob = probs
    neg_prob = 1.0 - probs
    if clip and clip > 0:
        neg_prob = (neg_prob + clip).clamp(max=1.0)

    ce = -(targets * torch.log(pos_prob + eps) + (1 - targets) * torch.log(neg_prob + eps))
    pt = targets * pos_prob + (1 - targets) * neg_prob
    gamma = targets * gamma_pos + (1 - targets) * gamma_neg
    modulator = (1 - pt).pow(gamma)
    loss = modulator * ce
    return loss.mean()


def tune_thresholds_fbeta(
    y_true: torch.Tensor,
    y_prob: torch.Tensor,
    *,
    beta: float = 0.5,
    min_threshold: float = 0.4,
) -> torch.Tensor:
    """Tune per-class thresholds with F-beta (precision-emphasised) objective."""

    grid = torch.linspace(0.05, 0.95, steps=19)
    num_classes = y_true.size(1)
    thresholds = torch.full((num_classes,), 0.5, dtype=torch.float32)
    beta_sq = beta * beta

    for c in range(num_classes):
        target = y_true[:, c].bool()
        scores = y_prob[:, c]
        best_score = -1.0
        best_thr = 0.5
        for thr in grid:
            pred = scores >= thr
            tp = (pred & target).sum().item()
            fp = (pred & ~target).sum().item()
            fn = (~pred & target).sum().item()
            denom = (1 + beta_sq) * tp + beta_sq * fn + fp
            score = 0.0 if denom == 0 else (1 + beta_sq) * tp / denom
            if score > best_score:
                best_score = score
                best_thr = float(thr)
        thresholds[c] = max(min_threshold, best_thr)

    return thresholds


def estimate_label_stats(
    loader, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute positive/negative counts, pos_weight, and class priors."""

    pos_total: Optional[torch.Tensor] = None
    num_samples = 0
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"]["binary"].to(device).float()
            pos_batch = labels.sum(dim=0)
            pos_total = pos_batch if pos_total is None else pos_total + pos_batch
            num_samples += labels.size(0)

    if pos_total is None:
        raise RuntimeError("Failed to compute label statistics (empty loader)")

    pos_total = pos_total.clamp_min(1.0)
    neg_total = (num_samples - pos_total).clamp_min(1.0)
    pos_weight = (neg_total / pos_total).float()
    prior = (pos_total / (pos_total + neg_total)).float().clamp_(1e-4, 1.0 - 1e-4)
    return pos_weight, pos_total, neg_total, prior


def run_epoch(
    model: GazeIntent2TranscriptAndLabels,
    loader,
    *,
    device: torch.device,
    vocab,
    optimiser: Optional[AdamW],
    desc: str = "",
    lambda_text: float,
    lambda_speak: float,
    pos_weight: Optional[torch.Tensor],
    thresholds: Optional[torch.Tensor],
    use_asl: bool,
    asl_gamma_pos: float,
    asl_gamma_neg: float,
    asl_clip: float,
) -> Tuple[float, torch.Tensor, int, Dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    train_mode = optimiser is not None
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_cases = 0
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    all_logits: List[torch.Tensor] = []
    pw = pos_weight.to(device) if pos_weight is not None else None

    progress = tqdm(loader, desc=desc or "epoch", leave=False)
    for batch in progress:
        fix = batch["fixations"]
        labels_batch = batch["labels"]["binary"].to(device)
        images = batch["images"].to(device)
        transcripts = batch["transcripts"]

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

            with torch.set_grad_enabled(train_mode):
                outs_lbl = model.forward_case(
                    fixations={
                        "xy": xy,
                        "dwell": dwell,
                        "time": time_s,
                        "seg_hits": seg_hits,
                        "box_hits": box_hits,
                    },
                    transcript=None,
                    encode_text_fn=lambda s: vocab.encode(s, add_eos=True),
                    image_1chw=image,
                )

                if use_asl:
                    loss_labels = asymmetric_focal_loss(
                        outs_lbl["label_logits"],
                        labels.float(),
                        gamma_pos=asl_gamma_pos,
                        gamma_neg=asl_gamma_neg,
                        clip=asl_clip,
                    )
                else:
                    loss_labels = F.binary_cross_entropy_with_logits(
                        outs_lbl["label_logits"],
                        labels.float(),
                        pos_weight=pw,
                    )

                loss_text = torch.tensor(0.0, device=device)
                loss_speak = torch.tensor(0.0, device=device)

                if model.use_text and isinstance(transcript, dict):
                    outs_txt = model.forward_case(
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

                    segments_iterable = transcript.get("segments", [])
                    tok_total = 0
                    for logits, seg in zip(outs_txt["txt_logits_per_segment"], segments_iterable):
                        if logits.numel() == 0:
                            continue
                        tgt = vocab.encode(seg.get("text", ""), add_eos=True).to(device)
                        if tgt.numel() < 2:
                            continue
                        loss_text = loss_text + F.cross_entropy(logits, tgt[1:], reduction="sum")
                        tok_total += tgt.numel() - 1
                    if tok_total > 0:
                        loss_text = loss_text / tok_total

                    if segments_iterable:
                        speak_tgt = torch.zeros_like(outs_txt["speak_logits"])
                        begins = [float(seg.get("begin", 0.0)) for seg in segments_iterable]
                        for begin in begins:
                            nearest_idx = int(torch.argmin(torch.abs(time_s - begin)).item())
                            speak_tgt[nearest_idx] = 1.0
                        loss_speak = F.binary_cross_entropy_with_logits(outs_txt["speak_logits"], speak_tgt)

                loss = loss_labels + lambda_text * loss_text + lambda_speak * loss_speak

            case_losses.append(loss)
            total_loss += float(loss.item())
            logits = outs_lbl["label_logits"].detach().cpu()
            preds_case.append(torch.sigmoid(logits))
            all_logits.append(logits)
            targets_case.append(labels.detach().cpu())
            total_cases += 1

        if not case_losses:
            continue

        loss_value = torch.stack(case_losses).mean()

        if train_mode:
            optimiser.zero_grad(set_to_none=True)
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

        all_preds.extend(preds_case)
        all_targets.extend(targets_case)

    if all_preds:
        pred_tensor = torch.stack(all_preds, dim=0)
        target_tensor = torch.stack(all_targets, dim=0)
        logit_tensor = torch.stack(all_logits, dim=0)
    else:
        pred_tensor = torch.zeros(1, model.label_head.out_features)
        target_tensor = torch.zeros_like(pred_tensor)
        logit_tensor = torch.zeros_like(pred_tensor)

    per_class_acc = compute_class_accuracy(pred_tensor, target_tensor, thresholds)
    avg_loss = total_loss / max(1, total_cases)
    summary_metrics = compute_multilabel_summary(pred_tensor, target_tensor, thresholds)
    return avg_loss, per_class_acc, total_cases, summary_metrics, pred_tensor, target_tensor, logit_tensor


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
    use_gaze: bool,
    pretrained_image: bool,
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
    label_names: List[str],
    epoch: int,
    config_path: Path,
    metrics: Dict[str, float],
    hparams: Dict[str, object],
    extra_state: Optional[Dict[str, object]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vocab_state = {"itos": vocab.itos}
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimiser.state_dict(),
        "vocab": vocab_state,
        "label_names": label_names,
        "config_path": str(config_path),
        "metrics": metrics,
        "hparams": hparams,
    }
    if extra_state:
        checkpoint.update(extra_state)
    torch.save(checkpoint, path)
    print(f"  Checkpoint saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train silence-thought RNN with pretrained image encoder")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "data_egd-cxr.yaml", help="Path to configuration file")
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
    parser.add_argument("--lambda-text", type=float, default=0.1, help="Weight for language loss")
    parser.add_argument("--lambda-speak", type=float, default=0.1, help="Weight for speak gate loss")
    parser.add_argument("--use-pos-weight", action="store_true", help="Apply per-class pos_weight in BCE loss")
    parser.add_argument("--tune-thresholds", action="store_true", help="Tune per-class thresholds on validation set")
    parser.add_argument("--pos-weight-clip", type=float, default=25.0, help="Clip pos_weight to this maximum (<=0 to disable)")
    parser.add_argument("--use-logit-adjust", action="store_true", help="Subtract logit priors at evaluation time")
    parser.add_argument("--use-asl", action="store_true", help="Use asymmetric focal loss for label prediction")
    parser.add_argument("--asl-gamma-pos", type=float, default=0.0, help="Gamma for positive examples in ASL")
    parser.add_argument("--asl-gamma-neg", type=float, default=4.0, help="Gamma for negative examples in ASL")
    parser.add_argument("--asl-clip", type=float, default=0.05, help="Probability clip for ASL negative branch")
    parser.add_argument("--pretrained-image", dest="pretrained_image", action="store_true", help="Use pretrained ResNet-18 image encoder")
    parser.add_argument("--no-pretrained-image", dest="pretrained_image", action="store_false", help="Disable pretrained image encoder")
    parser.set_defaults(pretrained_image=True)
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
            "lambda_text": train_cfg.get("lambda_text"),
            "lambda_speak": train_cfg.get("lambda_speak"),
            "pos_weight_clip": train_cfg.get("pos_weight_clip"),
            "asl_gamma_pos": train_cfg.get("asl_gamma_pos"),
            "asl_gamma_neg": train_cfg.get("asl_gamma_neg"),
            "asl_clip": train_cfg.get("asl_clip"),
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
            "use_gaze": train_cfg.get("use_gaze"),
            "use_pos_weight": train_cfg.get("use_pos_weight"),
            "tune_thresholds": train_cfg.get("tune_thresholds"),
            "use_logit_adjust": train_cfg.get("use_logit_adjust"),
            "use_asl": train_cfg.get("use_asl"),
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

    print("Creating data loaders...")
    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = create_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = create_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    pos_weight: Optional[torch.Tensor] = None
    logit_adjust: Optional[torch.Tensor] = None
    if args.use_pos_weight or args.use_logit_adjust:
        print("Estimating label statistics from training data...")
        stats_loader = create_dataloader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        pos_weight_raw, pos_total, neg_total, pos_prior = estimate_label_stats(stats_loader, device)

        if args.use_pos_weight:
            pos_weight = pos_weight_raw.sqrt()
            if args.pos_weight_clip > 0:
                pos_weight = pos_weight.clamp(max=args.pos_weight_clip)
            print(
                f"  pos_weight mean {float(pos_weight.mean()):.2f} | min "
                f"{float(pos_weight.min()):.2f} | max {float(pos_weight.max()):.2f}"
            )

        if args.use_logit_adjust:
            logit_adjust = torch.log(pos_prior / (1 - pos_prior))
            print(
                f"  logit adjustment mean {float(logit_adjust.mean()):.3f} | "
                f"median {float(logit_adjust.median()):.3f}"
            )
    else:
        logit_adjust = None

    if args.use_asl:
        pos_weight = None

    logit_adjust_cpu = logit_adjust.cpu() if logit_adjust is not None else None
    if logit_adjust_cpu is not None:
        logit_adjust_cpu = logit_adjust_cpu.to(torch.float32)

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
    label_names = train_dataset.label_proc.schema.class_columns
    print(f"Training on {len(label_names)} disease classes: {label_names}")

    print(f"\nStarting training for {args.epochs} epochs...")
    checkpoint_dir = args.checkpoint_dir
    run_id = time.strftime("%Y%m%d-%H%M%S")
    best_val_loss = float("inf")
    best_checkpoint_path: Optional[Path] = None
    last_train_loss = float("nan")
    last_val_loss = float("nan")
    train_summary = {"macro_acc": 0.0, "macro_f1": 0.0, "micro_f1": 0.0, "micro_precision": 0.0, "micro_recall": 0.0, "macro_pr_auc": 0.0, "micro_pr_auc": 0.0}
    val_summary = {"macro_acc": 0.0, "macro_f1": 0.0, "micro_f1": 0.0, "micro_precision": 0.0, "micro_recall": 0.0, "macro_pr_auc": 0.0, "micro_pr_auc": 0.0}
    current_thresholds: Optional[torch.Tensor] = None
    best_thresholds: Optional[torch.Tensor] = None

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
        "lambda_text": args.lambda_text,
        "lambda_speak": args.lambda_speak,
        "use_pos_weight": args.use_pos_weight,
        "tune_thresholds": args.tune_thresholds,
        "pos_weight_clip": args.pos_weight_clip,
        "use_logit_adjust": args.use_logit_adjust,
        "use_asl": args.use_asl,
        "asl_gamma_pos": args.asl_gamma_pos,
        "asl_gamma_neg": args.asl_gamma_neg,
        "asl_clip": args.asl_clip,
    }

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, _, _, _, train_probs, train_targets, train_logits = run_epoch(
            model,
            train_loader,
            device=device,
            vocab=vocab,
            optimiser=optimiser,
            desc=f"train {epoch:02d}",
            lambda_text=args.lambda_text,
            lambda_speak=args.lambda_speak,
            pos_weight=pos_weight,
            thresholds=current_thresholds,
            use_asl=args.use_asl,
            asl_gamma_pos=args.asl_gamma_pos,
            asl_gamma_neg=args.asl_gamma_neg,
            asl_clip=args.asl_clip,
        )

        val_loss, _, _, _, val_probs, val_targets, val_logits = run_epoch(
            model,
            val_loader,
            device=device,
            vocab=vocab,
            optimiser=None,
            desc=f"val {epoch:02d}",
            lambda_text=args.lambda_text,
            lambda_speak=args.lambda_speak,
            pos_weight=pos_weight,
            thresholds=current_thresholds,
            use_asl=args.use_asl,
            asl_gamma_pos=args.asl_gamma_pos,
            asl_gamma_neg=args.asl_gamma_neg,
            asl_clip=args.asl_clip,
        )

        train_logits = train_logits.to(train_probs.device, dtype=train_probs.dtype) if isinstance(train_logits, torch.Tensor) else train_logits
        val_logits = val_logits.to(val_probs.device, dtype=val_probs.dtype) if isinstance(val_logits, torch.Tensor) else val_logits

        if args.use_logit_adjust and logit_adjust_cpu is not None:
            adjust = logit_adjust_cpu.to(train_logits.device, dtype=train_logits.dtype)
            train_probs_adj = torch.sigmoid(train_logits - adjust)
            val_probs_adj = torch.sigmoid(val_logits - adjust)
        else:
            train_probs_adj = train_probs
            val_probs_adj = val_probs

        thresholds_for_metrics = current_thresholds
        if args.tune_thresholds:
            thresholds_for_metrics = tune_thresholds_fbeta(val_targets, val_probs_adj, beta=0.5)
            current_thresholds = thresholds_for_metrics.clone()

        train_summary = compute_multilabel_summary(train_probs_adj, train_targets, thresholds_for_metrics)
        train_acc = compute_class_accuracy(train_probs_adj, train_targets, thresholds_for_metrics)
        val_summary = compute_multilabel_summary(val_probs_adj, val_targets, thresholds_for_metrics)
        val_acc = compute_class_accuracy(val_probs_adj, val_targets, thresholds_for_metrics)

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
        if thresholds_for_metrics is not None:
            metrics["val_thresholds"] = [float(x) for x in thresholds_for_metrics.tolist()]
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_checkpoint_path = checkpoint_dir / f"{run_id}_best.pt"
            best_thresholds = thresholds_for_metrics.clone() if thresholds_for_metrics is not None else None
            save_checkpoint(
                best_checkpoint_path,
                model=model,
                optimiser=optimiser,
                vocab=vocab,
                label_names=label_names,
                epoch=epoch,
                config_path=args.config,
                metrics=metrics,
                hparams=hparams,
                extra_state={
                    "val_thresholds": [float(x) for x in thresholds_for_metrics.tolist()] if thresholds_for_metrics is not None else None,
                },
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
        if thresholds_for_metrics is not None:
            pos_rate_default = float((val_probs_adj >= 0.5).float().mean().item())
            pos_rate_thr = float((val_probs_adj >= thresholds_for_metrics).float().mean().item())
            print(
                f"  Val threshold stats | median {thresholds_for_metrics.median():.3f} | "
                f"pos-rate@0.50 {pos_rate_default:.3f} | pos-rate@thr {pos_rate_thr:.3f}"
            )

    print("\nEvaluating on test set...")
    test_thresholds = best_thresholds if best_thresholds is not None else current_thresholds
    test_loss, _, batches, _, test_probs, test_targets, test_logits = run_epoch(
        model,
        test_loader,
        device=device,
        vocab=vocab,
        optimiser=None,
        desc="test",
        lambda_text=args.lambda_text,
        lambda_speak=args.lambda_speak,
        pos_weight=pos_weight,
        thresholds=test_thresholds,
        use_asl=args.use_asl,
        asl_gamma_pos=args.asl_gamma_pos,
        asl_gamma_neg=args.asl_gamma_neg,
        asl_clip=args.asl_clip,
    )
    test_logits = test_logits.to(test_probs.device, dtype=test_probs.dtype)
    if args.use_logit_adjust and logit_adjust_cpu is not None:
        adjust = logit_adjust_cpu.to(test_logits.device, dtype=test_logits.dtype)
        test_probs_adj = torch.sigmoid(test_logits - adjust)
    else:
        test_probs_adj = test_probs
    if test_thresholds is not None:
        test_summary = compute_multilabel_summary(test_probs_adj, test_targets, test_thresholds)
        test_acc = compute_class_accuracy(test_probs_adj, test_targets, test_thresholds)
    else:
        test_summary = compute_multilabel_summary(test_probs_adj, test_targets, None)
        test_acc = compute_class_accuracy(test_probs_adj, test_targets, None)
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
    final_thresholds = best_thresholds if best_thresholds is not None else current_thresholds
    final_thresholds_cpu = final_thresholds.clone().cpu() if final_thresholds is not None else None
    save_checkpoint(
        final_checkpoint_path,
        model=model,
        optimiser=optimiser,
        vocab=vocab,
        label_names=label_names,
        epoch=args.epochs,
        config_path=args.config,
        metrics=final_metrics,
        hparams=hparams,
        extra_state={
            "val_thresholds": [float(x) for x in final_thresholds_cpu.tolist()] if final_thresholds_cpu is not None else None,
        },
    )
    if best_checkpoint_path is not None and best_checkpoint_path != final_checkpoint_path:
        print(f"\nBest validation checkpoint: {best_checkpoint_path}")
    print(f"Final checkpoint: {final_checkpoint_path}")


if __name__ == "__main__":
    main()
