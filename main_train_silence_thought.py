#!/usr/bin/env python3
"""Train the silence-thought RNN on EGD-CXR gaze data."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from egd_cxr_dataset import ConfigLoader, EGDCXRDataset, build_vocab, create_dataloader
from egd_cxr_dataset.models import GazeIntent2TranscriptAndLabels


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


def compute_class_accuracy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bin_pred = (pred >= 0.5).float()
    bin_target = (target >= 0.5).float()
    correct = (bin_pred == bin_target).float()
    return correct.mean(dim=0)


def run_epoch(
    model: GazeIntent2TranscriptAndLabels,
    loader,
    *,
    device: torch.device,
    vocab,
    optimiser: Optional[AdamW],
) -> Tuple[float, torch.Tensor, int]:
    train_mode = optimiser is not None
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_cases = 0
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    for batch in loader:
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

            loss_labels = F.binary_cross_entropy_with_logits(outputs["label_logits"], labels.float())
            loss_text = torch.tensor(0.0, device=device)
            for logits, seg in zip(outputs["txt_logits_per_segment"], transcript.get("segments", [])):
                if logits.numel() == 0:
                    continue
                tgt = vocab.encode(seg.get("text", ""), add_eos=True).to(device)
                loss_text = loss_text + F.cross_entropy(logits, tgt, reduction="mean")
            loss = loss_labels + loss_text

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
    return avg_loss, per_class_acc, total_cases


def build_model_and_vocab(
    train_dataset: EGDCXRDataset,
    device: torch.device,
    *,
    txt_dim: int,
    enc_dim: int,
    max_decode_len: int,
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
    ).to(device)

    return model, vocab


def format_accuracy(label_names: List[str], acc: torch.Tensor) -> str:
    pieces = [f"{name}: {float(a):.3f}" for name, a in zip(label_names, acc.tolist())]
    return ", ".join(pieces)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train silence-thought RNN on EGD-CXR gaze data")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "data_egd-cxr.yaml")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-fixations", type=int, default=None)
    parser.add_argument("--txt-dim", type=int, default=256)
    parser.add_argument("--enc-dim", type=int, default=256)
    parser.add_argument("--rnn-hidden", type=int, default=256)
    parser.add_argument("--rnn-layers", type=int, default=2)
    parser.add_argument("--max-decode-len", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_loader = ConfigLoader(args.config)
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

    train_dataset = EGDCXRDataset(
        root=gaze_root,
        seg_path=seg_dir,
        transcripts_path=transcripts_dir,
        dicom_root=dicom_root,
        max_fixations=args.max_fixations,
        case_ids=train_ids,
    )
    val_dataset = EGDCXRDataset(
        root=gaze_root,
        seg_path=seg_dir,
        transcripts_path=transcripts_dir,
        dicom_root=dicom_root,
        max_fixations=args.max_fixations,
        case_ids=val_ids,
    )
    test_dataset = EGDCXRDataset(
        root=gaze_root,
        seg_path=seg_dir,
        transcripts_path=transcripts_dir,
        dicom_root=dicom_root,
        max_fixations=args.max_fixations,
        case_ids=test_ids,
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = create_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(">>>>>>>>>>>>>> print sample here")
    sample = train_dataset[0]  # or train_dataset[0], val_dataset[0], etc.
    print("sample keys:", list(sample.keys()))
    for k, v in sample.items():
        try:
            print(k, type(v), getattr(v, "shape", None))
        except Exception:
            print(k, type(v)) 
    print("<<<<<<<<<<<<<< print sample here")
    print("================================================") 
    batch = next(iter(train_loader))  # or train_loader/val_loader
    print("batch keys:", list(batch.keys()))

    # Inspect the first item in the batch
    idx = 0
    first_item = {k: (v[idx] if hasattr(v, "__getitem__") else v) for k, v in batch.items()}
    print("first item keys:", list(first_item.keys()))
    for k, v in first_item.items():
        try:
            print(k, type(v), getattr(v, "shape", None))
        except Exception:
            print(k, type(v)) 
    print("================================================") 

    model, vocab = build_model_and_vocab(
        train_dataset,
        device,
        txt_dim=args.txt_dim,
        enc_dim=args.enc_dim,
        max_decode_len=args.max_decode_len,
    )

    optimiser = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    label_names = train_dataset.label_proc.schema.class_columns

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _ = run_epoch(
            model,
            train_loader,
            device=device,
            vocab=vocab,
            optimiser=optimiser,
        )
        val_loss, val_acc, _ = run_epoch(
            model,
            val_loader,
            device=device,
            vocab=vocab,
            optimiser=None,
        )
        print(f"Epoch {epoch:02d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")
        print("  Train accuracy per class:")
        print("    " + format_accuracy(label_names, train_acc))
        print("  Val accuracy per class:")
        print("    " + format_accuracy(label_names, val_acc))

    test_loss, test_acc, batches = run_epoch(
        model,
        test_loader,
        device=device,
        vocab=vocab,
        optimiser=None,
    )
    print(f"Test loss {test_loss:.4f} over {batches} batches")
    print("Test accuracy per class:")
    print("  " + format_accuracy(label_names, test_acc))

    # Show a quick qualitative prediction
    batch = next(iter(test_loader))
    fix = batch["fixations"]
    length = int(fix["lengths"][0].item())
    case = {
        "xy": fix["xy"][0, :length].to(device),
        "dwell": fix["dwell"][0, :length].to(device),
        "time": fix["time"][0, :length].to(device),
        "seg_hits": fix["seg_hits"][0, :length].to(device),
        "box_hits": fix["box_hits"][0, :length].to(device),
    }
    outputs = model.generate_case(
        fixations=case,
        transcript=batch["transcripts"][0],
        encode_text_fn=lambda s: vocab.encode(s, add_eos=True),
        image_1chw=batch["images"][0].to(device),
        max_len=args.max_decode_len,
    )
    decoded_segments = [vocab.decode(tokens.tolist()) for tokens in outputs["gen_tokens_per_segment"]]
    print("Sample prediction:")
    print(
        json.dumps(
            {
                "dicom_id": batch["dicom_ids"][0],
                "label_probs": torch.sigmoid(outputs["label_logits"]).cpu().tolist(),
                "segments": decoded_segments,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
