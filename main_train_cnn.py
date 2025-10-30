#!/usr/bin/env python3
"""
main_train_cnn.py
Train ImageCNN (image only, no RNN/gaze/text modalities).
Backbones: resnet18, resnet50, densenet121, txrv_densenet121.
Prints and saves best model by val loss. Shows tqdm progress bars.
"""
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import argparse
import sys 
# ==== Ensure local src import works (fix for ModuleNotFoundError) =====
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# =====================================================================

# ------------------------------------------------------------------ #
# Dataset & model
# ------------------------------------------------------------------ #
 
from src.models.image_cnn import ImageCNN
from egd_cxr_dataset.datasets.egd_cxr import EGDCXRDataset

def set_seed(seed: int = 2025):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_epoch(model, loader, device, criterion, optimizer=None, desc="train"):
    if optimizer is not None:
        model.train()
    else:
        model.eval()
    total_loss, total_correct, total_count = 0, 0, 0
    pbar = tqdm(loader, desc=desc)
    for batch in pbar:
        y = batch["labels"]["single_index"].to(device)
        with torch.set_grad_enabled(optimizer is not None):
            logits, _ = model(batch)
            loss = criterion(logits, y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            preds = logits.argmax(1)
            correct = (preds == y).sum().item()
            total_correct += correct
            total_loss += loss.item() * y.size(0)
            total_count += y.size(0)
        pbar.set_postfix(loss=total_loss / total_count, acc=100 * total_correct / total_count)
    return total_loss / total_count, total_correct / total_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image_backbone", type=str, default=None)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()
    set_seed(args.seed)
    cfg = yaml.safe_load(args.config.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(cfg["train"]["classes"])
    image_backbone = args.image_backbone or cfg.get("image_backbone") or cfg.get("options", {}).get("image_backbone") or "resnet50"
    d_img = cfg["model"].get("d_img", 128)
    use_proj = True
    model = ImageCNN(num_classes=num_classes, image_backbone=image_backbone, d_img=d_img, use_proj=use_proj).to(device)
    # train_set = EGDCXRDataset(cfg, split="train")  # (implement this as in your main_train_v2.py)
    train_set = EGDCXRDataset(..., case_ids=train_ids, ...)
    val_set = EGDCXRDataset(cfg, split="val")
    test_set = EGDCXRDataset(cfg, split="test")
    class_counts = train_set.class_counts if hasattr(train_set, "class_counts") else None
    if class_counts is not None:
        weights = [1.0 / (class_counts[y] + 1e-6) for y in train_set.labels]
        sampler = WeightedRandomSampler(weights, len(train_set))
    else:
        sampler = None
    train_loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size"], sampler=sampler, shuffle=(sampler is None))
    val_loader = DataLoader(val_set, batch_size=cfg["train"]["batch_size"])
    test_loader = DataLoader(test_set, batch_size=cfg["train"]["batch_size"])
    optimizer = AdamW(model.parameters(), lr=cfg["train"].get("lr", 3e-4), weight_decay=cfg["train"].get("weight_decay", 1e-4))
    criterion = nn.CrossEntropyLoss()
    epochs = cfg["train"]["epochs"]
    best_loss = float('inf')
    best_path = Path(cfg["output_path"]["checkpoint_dir"]) / (cfg["output_path"].get("best_ckpt_name") or "cnn_best.pt")
    best_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train_loss, train_acc = run_epoch(model, train_loader, device, criterion, optimizer, f"train E{epoch}")
        val_loss, val_acc = run_epoch(model, val_loader, device, criterion, None, f"val   E{epoch}")
        print(f"E{epoch}: train loss={train_loss:.4f} acc={train_acc:.4f}   val loss={val_loss:.4f} acc={val_acc:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"  [BEST] Model saved → {best_path}")
    print("Evaluating best model on test set...")
    model.load_state_dict(torch.load(best_path))
    test_loss, test_acc = run_epoch(model, test_loader, device, criterion, None, "test best")
    print(f"Test: loss={test_loss:.4f}, acc={test_acc:.4f}")

if __name__ == "__main__":
    main()