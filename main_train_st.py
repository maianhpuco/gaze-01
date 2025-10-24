#!/usr/bin/env python3
"""
Train the silence-thought RNN on EGD-CXR gaze data.

This script implements a multimodal machine learning pipeline that trains a neural network
to predict radiologist transcripts and diagnostic labels from eye-gaze tracking data on
chest X-ray images. The model learns to map gaze patterns to clinical findings.

Key Components:
- Eye-gaze fixation sequences (coordinates, timing, duration)
- Anatomical region mappings (which body parts were looked at)
- Bounding box annotations (abnormalities detected)
- Radiologist transcripts (what was said during examination)
- Diagnostic labels (binary disease classifications)

The model architecture combines:
- RNN for processing sequential gaze data
- CNN for processing chest X-ray images
- Attention mechanisms for focusing on relevant regions
- Multi-task learning (transcript generation + disease classification)
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

# Set up project paths for imports
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Import custom dataset and model classes
from egd_cxr_dataset import ConfigLoader, EGDCXRDataset, build_vocab, create_dataloader
from egd_cxr_dataset.models.gaze_intent import GazeIntent2TranscriptAndLabels


def set_seed(seed: int = 0) -> None:
    """
    Set random seeds for reproducibility across all random number generators.
    
    This ensures that experiments can be reproduced exactly by using the same seed.
    Important for scientific reproducibility in machine learning experiments.
    
    Args:
        seed: Random seed value (default: 0)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_split_ids(split_dir: Path, split: str) -> List[str]:
    """
    Load case IDs for a specific data split (train/val/test).
    
    The dataset is pre-split into train/validation/test sets to ensure consistent
    evaluation. Each split file contains one case ID per line.
    
    Args:
        split_dir: Directory containing split files
        split: Split name ('train', 'val', or 'test')
        
    Returns:
        List of case IDs for the specified split
        
    Raises:
        FileNotFoundError: If the split file doesn't exist
    """
    path = split_dir / f"{split}_ids.txt"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def gather_transcripts(dataset: EGDCXRDataset, case_ids: Iterable[str]) -> List[str]:
    """
    Extract transcript text from the dataset for vocabulary building.
    
    This function collects all radiologist transcripts from the training cases
    to build a vocabulary for text generation. The transcripts contain the spoken
    observations during X-ray examination.
    
    Args:
        dataset: The EGD-CXR dataset instance
        case_ids: List of case IDs to extract transcripts from
        
    Returns:
        List of transcript text strings
    """
    texts: List[str] = []
    for case_id in case_ids:
        payload = dataset._load_transcript(case_id)  # pylint: disable=protected-access
        texts.append(payload.get("text", ""))
    return texts


def compute_class_accuracy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute per-class accuracy for multi-label binary classification.
    
    This function evaluates how well the model predicts each disease/condition
    by comparing predicted probabilities (>=0.5 threshold) with ground truth labels.
    
    Args:
        pred: Model predictions (probabilities) [batch_size, num_classes]
        target: Ground truth labels [batch_size, num_classes]
        
    Returns:
        Per-class accuracy tensor [num_classes]
    """
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
    desc: str = "",
) -> Tuple[float, torch.Tensor, int]:
    """
    Run a single training or evaluation epoch.
    
    This function processes all batches in the data loader, computing losses and
    predictions for both transcript generation and disease classification tasks.
    
    Args:
        model: The neural network model to train/evaluate
        loader: DataLoader providing batches of multimodal data
        device: PyTorch device (CPU/GPU)
        vocab: Vocabulary for text encoding/decoding
        optimiser: Optimizer for training (None for evaluation)
        desc: Description for progress bar
        
    Returns:
        Tuple of (average_loss, per_class_accuracy, total_cases_processed)
    """
    # Determine if we're in training or evaluation mode
    train_mode = optimiser is not None
    if train_mode:
        model.train()
    else:
        model.eval()

    # Initialize tracking variables
    total_loss = 0.0
    total_cases = 0
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    # Process each batch with progress tracking
    progress = tqdm(loader, desc=desc or "epoch", leave=False)
    for batch in progress:
        # Extract multimodal data from batch
        fix = batch["fixations"]                    # Eye tracking data (coordinates, timing, etc.)
        labels_batch = batch["labels"]["binary"].to(device)  # Disease labels (binary classification)
        images = batch["images"].to(device)          # Chest X-ray images
        transcripts = batch["transcripts"]          # Radiologist speech transcripts

        # Initialize batch-level tracking
        batch_loss = 0.0
        case_losses: List[torch.Tensor] = []
        preds_case: List[torch.Tensor] = []
        targets_case: List[torch.Tensor] = []

        # Process each case in the batch individually
        for idx in range(labels_batch.size(0)):
            # Get sequence length for this case (variable-length sequences)
            length = int(fix["lengths"][idx].item())
            if length == 0:
                continue  # Skip cases with no gaze data

            # Extract gaze data for this case (trimmed to actual length)
            xy = fix["xy"][idx, :length].to(device)           # Gaze coordinates (x, y)
            dwell = fix["dwell"][idx, :length].to(device)     # Fixation durations (ms)
            time_s = fix["time"][idx, :length].to(device)    # Timestamps (seconds)
            seg_hits = fix["seg_hits"][idx, :length].to(device)  # Which anatomical regions were looked at
            box_hits = fix["box_hits"][idx, :length].to(device)  # Which abnormalities were looked at
            
            # Get other modalities for this case
            image = images[idx]                    # Chest X-ray image
            transcript = transcripts[idx]          # Radiologist transcript
            labels = labels_batch[idx]             # Disease labels

            # Forward pass through the model
            outputs = model.forward_case(
                fixations={
                    "xy": xy,                    # Gaze coordinates
                    "dwell": dwell,              # Fixation durations
                    "time": time_s,              # Timestamps
                    "seg_hits": seg_hits,        # Anatomical region hits
                    "box_hits": box_hits,        # Abnormality hits
                },
                transcript=transcript,           # Ground truth transcript
                encode_text_fn=lambda s: vocab.encode(s, add_eos=True),  # Text encoding function
                image_1chw=image,                # Chest X-ray image
            )

            # Compute loss for disease classification task
            loss_labels = F.binary_cross_entropy_with_logits(outputs["label_logits"], labels.float())
            
            # Compute loss for transcript generation task (if segments exist)
            loss_text = torch.tensor(0.0, device=device)
            tok_total = 0
            if model.use_text:
                segments_iterable = transcript.get("segments", [])
                for logits, seg in zip(outputs["txt_logits_per_segment"], segments_iterable):
                    if logits.numel() == 0:
                        continue
                    tgt = vocab.encode(seg.get("text", ""), add_eos=True).to(device)
                    loss_text = loss_text + F.cross_entropy(logits, tgt, reduction="sum")
                    tok_total += tgt.numel()
                if tok_total > 0:
                    loss_text = loss_text / tok_total
            
            # Combined loss (multi-task learning)
            loss = loss_labels + loss_text

            # Accumulate loss and predictions for this case
            batch_loss += loss
            case_losses.append(loss)
            preds_case.append(torch.sigmoid(outputs["label_logits"]).detach().cpu())  # Convert logits to probabilities
            targets_case.append(labels.detach().cpu())
            total_cases += 1

        # Skip batch if no valid cases
        if not case_losses:
            continue

        # Compute average loss for this batch
        loss_value = batch_loss / len(case_losses)
        
        # Training step (backpropagation and optimization)
        if train_mode:
            optimiser.zero_grad(set_to_none=True)  # Clear gradients
            loss_value.backward()                  # Compute gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimiser.step()                       # Update model parameters

        # Accumulate results for epoch-level metrics
        total_loss += float(loss_value.item()) * len(case_losses)
        all_preds.extend(preds_case)
        all_targets.extend(targets_case)

    # Compute epoch-level metrics
    if all_preds:
        pred_tensor = torch.stack(all_preds, dim=0)      # Stack all predictions
        target_tensor = torch.stack(all_targets, dim=0)  # Stack all targets
    else:
        # Handle edge case with no valid predictions
        pred_tensor = torch.zeros(1, model.label_head.out_features)
        target_tensor = torch.zeros_like(pred_tensor)

    # Compute per-class accuracy and average loss
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
    use_bbox: bool,
    use_seg: bool,
    use_image: bool,
    use_text: bool,
):
    """
    Build the neural network model and vocabulary from training data.
    
    This function creates the multimodal model architecture and builds a vocabulary
    from the training transcripts. The model combines gaze data, images, and text
    for joint prediction of transcripts and disease labels.
    
    Args:
        train_dataset: Training dataset to extract metadata from
        device: PyTorch device for model placement
        txt_dim: Dimension for text decoder
        enc_dim: Dimension for encoder features
        max_decode_len: Maximum sequence length for text generation
        use_bbox: Whether to include bounding box features
        use_seg: Whether to include segmentation features
        use_image: Whether to include image features
        use_text: Whether to enable transcript decoder
        
    Returns:
        Tuple of (model, vocabulary)
    """
    # Build vocabulary from training transcripts
    train_texts = gather_transcripts(train_dataset, train_dataset.case_ids)
    vocab = build_vocab(train_texts, min_freq=1, max_size=30000)

    # Extract dataset dimensions from a sample
    sample = train_dataset[0]
    fix = sample["fixations"]
    num_segments = fix["seg_hits"].shape[1]        # Number of anatomical regions
    num_box_classes = fix["box_hits"].shape[1]     # Number of abnormality classes
    num_labels = sample["labels"]["binary"].shape[0]  # Number of disease labels

    # Create the multimodal model
    model = GazeIntent2TranscriptAndLabels(
        num_box_classes=num_box_classes,    # Number of abnormality types
        num_segments=num_segments,          # Number of anatomical regions
        img_out_dim=enc_dim,               # Image encoder output dimension
        intent_dim=enc_dim,                 # Intent encoder dimension
        vocab_size=vocab.size,             # Vocabulary size for text generation
        dec_dim=txt_dim,                   # Text decoder dimension
        num_labels=num_labels,             # Number of disease classification labels
        pad_id=vocab.pad_id,               # Padding token ID
        bos_id=vocab.bos_id,               # Beginning of sequence token ID
        eos_id=vocab.eos_id,               # End of sequence token ID
        use_box=use_bbox,
        use_seg=use_seg,
        use_image=use_image,
        use_text=use_text,
    ).to(device)

    return model, vocab


def format_accuracy(label_names: List[str], acc: torch.Tensor) -> str:
    """
    Format per-class accuracy results for display.
    
    Args:
        label_names: List of disease/condition names
        acc: Per-class accuracy tensor
        
    Returns:
        Formatted string showing accuracy for each class
    """
    pieces = [f"{name}: {float(a):.3f}" for name, a in zip(label_names, acc.tolist())]
    return ", ".join(pieces)


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
) -> None:
    """
    Persist model, optimiser, and vocabulary state to disk.
    """
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
    torch.save(checkpoint, path)
    print(f"  Checkpoint saved to {path}")


def main() -> None:
    """
    Main training function for the silence-thought model.
    
    This function orchestrates the entire training pipeline:
    1. Parse command-line arguments
    2. Set up data loaders for train/val/test splits
    3. Build model and vocabulary
    4. Train the model for specified epochs
    5. Evaluate on test set
    6. Show sample predictions
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train silence-thought RNN on EGD-CXR gaze data")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "data_egd-cxr.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--batch-size", type=int, default=4, 
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, 
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, 
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, 
                       help="Weight decay for regularization")
    parser.add_argument("--max-fixations", type=int, default=None, 
                       help="Maximum number of fixations per case (None for all)")
    parser.add_argument("--txt-dim", type=int, default=256, 
                       help="Text decoder dimension")
    parser.add_argument("--enc-dim", type=int, default=256, 
                       help="Encoder dimension")
    parser.add_argument("--rnn-hidden", type=int, default=256, 
                       help="RNN hidden dimension")
    parser.add_argument("--rnn-layers", type=int, default=2, 
                       help="Number of RNN layers")
    parser.add_argument("--max-decode-len", type=int, default=64, 
                       help="Maximum text generation length")
    parser.add_argument("--num-workers", type=int, default=0, 
                       help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=0, 
                       help="Random seed for reproducibility")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "runs" / "checkpoints", 
                       help="Directory to store model checkpoints")
    parser.add_argument("--no-bbox", action="store_true", help="Disable bounding box features")
    parser.add_argument("--no-seg", action="store_true", help="Disable segmentation region features")
    parser.add_argument("--no-image", action="store_true", help="Disable image features")
    parser.add_argument("--no-text", action="store_true", help="Disable transcript generation branch")
    args = parser.parse_args()

    # Derive feature toggles
    args.use_bbox = not args.no_bbox
    args.use_seg = not args.no_seg
    args.use_image = not args.no_image
    args.use_text = not args.no_text

    # Load configuration from disk
    config_loader = ConfigLoader(args.config)

    # Allow YAML config to override CLI defaults for training hyperparameters
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
            "rnn_hidden": train_cfg.get("rnn_hidden"),
            "rnn_layers": train_cfg.get("rnn_layers"),
            "max_decode_len": train_cfg.get("max_decode_len"),
            "num_workers": train_cfg.get("num_workers"),
            "seed": train_cfg.get("seed"),
            "checkpoint_dir": train_cfg.get("checkpoint_dir"),
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

    # Normalise checkpoint directory to a Path
    args.checkpoint_dir = Path(args.checkpoint_dir)
    if not args.checkpoint_dir.is_absolute():
        args.checkpoint_dir = ROOT / args.checkpoint_dir

    # Set up reproducibility and device
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up data paths using configuration
    gaze_root = Path(config_loader.get("input_path", "gaze_raw"))
    seg_dir = Path(config_loader.get("input_path", "segmentation_dir"))
    transcripts_dir = Path(config_loader.get("input_path", "transcripts_dir", default=seg_dir))
    dicom_root = Path(config_loader.get("input_path", "dicom_raw"))

    # Set up data splits
    split_dir_cfg = config_loader.get("split_files", "dir", default=ROOT / "config" / "splits")
    split_dir = Path(split_dir_cfg)
    if not split_dir.is_absolute():
        split_dir = ROOT / split_dir

    # Load case IDs for each split
    train_ids = read_split_ids(split_dir, "train")
    val_ids = read_split_ids(split_dir, "val")
    test_ids = read_split_ids(split_dir, "test")
    print(f"Data splits - Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    # Create datasets for each split
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

    # Create data loaders for each split
    print("Creating data loaders...")
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
    
    # Build model and vocabulary
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
    )
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Vocabulary size: {vocab.size}")

    # Set up optimizer and get label names
    optimiser = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    label_names = train_dataset.label_proc.schema.class_columns
    print(f"Training on {len(label_names)} disease classes: {label_names}")

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    checkpoint_dir = args.checkpoint_dir
    run_id = time.strftime("%Y%m%d-%H%M%S")
    best_val_loss = float("inf")
    best_checkpoint_path: Optional[Path] = None
    last_train_loss = float("nan")
    last_val_loss = float("nan")

    # Capture hyperparameters for checkpoint metadata
    hparams = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_fixations": args.max_fixations,
        "txt_dim": args.txt_dim,
        "enc_dim": args.enc_dim,
        "rnn_hidden": args.rnn_hidden,
        "rnn_layers": args.rnn_layers,
        "max_decode_len": args.max_decode_len,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "use_bbox": args.use_bbox,
        "use_seg": args.use_seg,
        "use_image": args.use_image,
        "use_text": args.use_text,
    }

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        # Training epoch
        train_loss, train_acc, _ = run_epoch(
            model,
            train_loader,
            device=device,
            vocab=vocab,
            optimiser=optimiser,
            desc=f"train {epoch:02d}",
        )
        
        # Validation epoch
        val_loss, val_acc, _ = run_epoch(
            model,
            val_loader,
            device=device,
            vocab=vocab,
            optimiser=None,
            desc=f"val {epoch:02d}",
        )
        epoch_duration = time.perf_counter() - epoch_start
        last_train_loss = train_loss
        last_val_loss = val_loss
        metrics = {
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "epoch_seconds": float(epoch_duration),
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
                label_names=label_names,
                epoch=epoch,
                config_path=args.config,
                metrics=metrics,
                hparams=hparams,
            )
        
        # Print epoch results
        print(
            f"Epoch {epoch:02d} | train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} | time {epoch_duration:.2f}s"
        )
        print("  Train accuracy per class:")
        print("    " + format_accuracy(label_names, train_acc))
        print("  Val accuracy per class:")
        print("    " + format_accuracy(label_names, val_acc))

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc, batches = run_epoch(
        model,
        test_loader,
        device=device,
        vocab=vocab,
        optimiser=None,
        desc="test",
    )
    print(f"Test loss {test_loss:.4f} over {batches} batches")
    print("Test accuracy per class:")
    print("  " + format_accuracy(label_names, test_acc))

    # Show a sample prediction for qualitative analysis when text decoder is enabled
    if model.use_text:
        print("\nGenerating sample prediction...")
        batch = next(iter(test_loader))
        fix = batch["fixations"]
        length = int(fix["lengths"][0].item())
        
        # Prepare gaze data for the first case in the batch
        case = {
            "xy": fix["xy"][0, :length].to(device),           # Gaze coordinates
            "dwell": fix["dwell"][0, :length].to(device),     # Fixation durations
            "time": fix["time"][0, :length].to(device),       # Timestamps
            "seg_hits": fix["seg_hits"][0, :length].to(device),  # Anatomical region hits
            "box_hits": fix["box_hits"][0, :length].to(device),  # Abnormality hits
        }
        
        # Generate predictions using the trained model
        outputs = model.generate_case(
            fixations=case,
            transcript=batch["transcripts"][0],
            encode_text_fn=lambda s: vocab.encode(s, add_eos=True),
            image_1chw=batch["images"][0].to(device),
            max_len=args.max_decode_len,
            min_len=min(args.max_decode_len, max(8, args.max_decode_len // 2)),
        )
        
        # Decode generated text segments
        decoded_segments = [vocab.decode(tokens.tolist()) for tokens in outputs["gen_tokens_per_segment"]]
        
        # Display sample prediction results
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
    else:
        print("\nTranscript decoder disabled; skipping sample text generation.")

    # Save final checkpoint snapshot
    final_checkpoint_path = checkpoint_dir / f"{run_id}_last.pt"
    final_metrics = {
        "train_loss": float(last_train_loss),
        "val_loss": float(last_val_loss),
        "test_loss": float(test_loss),
        "test_accuracy_per_class": {
            name: float(val) for name, val in zip(label_names, test_acc.tolist())
        },
    }
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
    )
    if best_checkpoint_path is not None and best_checkpoint_path != final_checkpoint_path:
        print(f"\nBest validation checkpoint: {best_checkpoint_path}")
    print(f"Final checkpoint: {final_checkpoint_path}")


if __name__ == "__main__":
    main()
