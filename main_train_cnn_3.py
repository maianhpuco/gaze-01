from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import sys
import yaml

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
from torchvision import transforms

# ----- local imports / sys.path fix -----
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from egd_cxr_dataset.datasets.egd_cxr import EGDCXRDataset, create_dataloader  # type: ignore
from src.models.image_cnn import ImageCNN

# Optional metrics (macro-AUC / per-class report)
try:
    from sklearn.metrics import roc_auc_score, classification_report
    HAS_SK = True
except Exception:
    HAS_SK = False


# ---------------- Helpers ----------------
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
        if s and not s.startswith("#"): ids.append(s)
    if not ids:
        raise FileNotFoundError(f"[ERROR] Split file exists but is empty: {path}")
    return ids

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    correct = (pred == targets).float().sum().item()
    return correct / max(1, targets.numel())


# --------- Epoch loops ----------
@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader,
    cfg: Dict[str, Any],
    device: torch.device,
) -> Tuple[float, float, Optional[float], Optional[str]]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss, total_acc, n = 0.0, 0.0, 0
    # collect for macro-AUC / report
    y_true_all, prob_all = [], []

    for batch in loader:
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)
            elif isinstance(v, dict):
                for sk, sv in v.items():
                    if isinstance(sv, torch.Tensor):
                        batch[k][sk] = sv.to(device, non_blocking=True)

        logits, _ = model(batch)
        y = batch["labels"]["single_index"].to(device).view(-1)
        loss = ce(logits, y)
        acc = accuracy(logits, y)

        bsz = y.size(0)
        total_loss += loss.item() * bsz
        total_acc  += acc * bsz
        n += bsz

        if HAS_SK:
            probs = torch.softmax(logits, dim=1).detach().cpu()
            prob_all.append(probs)
            y_true_all.append(y.detach().cpu())

    macro_auc = None
    cls_report = None
    if HAS_SK and n > 0:
        probs = torch.cat(prob_all, dim=0).numpy()
        y_true = torch.cat(y_true_all, dim=0).numpy()
        try:
            macro_auc = roc_auc_score(y_true, probs, multi_class="ovr")
        except Exception:
            macro_auc = None
        # per-class precision/recall/f1 (short string)
        preds = probs.argmax(axis=1)
        cls_report = classification_report(
            y_true, preds, digits=3, target_names=[str(c) for c in cfg["train"]["classes"]]
        )

    return total_loss / max(1, n), total_acc / max(1, n), macro_auc, cls_report


def run_one_epoch(
    model: nn.Module,
    loader,
    cfg: Dict[str, Any],
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    train: bool = True,
) -> Tuple[float, float]:
    model.train(train)
    ce = nn.CrossEntropyLoss(label_smoothing=float(cfg["train"].get("label_smoothing", 0.0)))
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    it = tqdm(loader, disable=not train, leave=False)
    for batch in it:
        # move tensors
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)
            elif isinstance(v, dict):
                for sk, sv in v.items():
                    if isinstance(sv, torch.Tensor):
                        batch[k][sk] = sv.to(device, non_blocking=True)

        logits, _ = model(batch)  # [B,C]
        y = batch["labels"]["single_index"].to(device).view(-1)

        loss = ce(logits, y)
        acc = accuracy(logits, y)

        if train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        bsz = y.size(0)
        total_loss += loss.item() * bsz
        total_acc  += acc * bsz
        n += bsz

    return total_loss / max(1, n), total_acc / max(1, n)


# ------------- Augmented Dataset Wrapper for Transforms ---------------
class AugmentedEGDCXRDataset(EGDCXRDataset):
    def __init__(self, transform=None, **kwargs):
        super().__init__(**kwargs)
        self.transform = transform

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        if self.transform and 'images' in item:
            # item["images"]: torch.Tensor [1,H,W] in [0,1]
            pil = transforms.ToPILImage(mode="L")(item["images"])
            img = self.transform(pil)  # transform should include ToTensor
            if img.ndim == 2:          # [H,W] -> [1,H,W]
                img = img.unsqueeze(0)
            elif img.size(0) == 3:     # if any op produced 3ch, collapse to 1ch
                img = img.mean(dim=0, keepdim=True)
            item["images"] = img
        return item


# ------------------ Main -----------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--image_backbone",
        type=str,
        default=None,
        choices=["resnet18", "resnet50", "densenet121", "txrv_densenet121"],
        help="Image encoder backbone override",
    )
    parser.add_argument("--no_proj", action="store_true", help="Disable projection head (classify from backbone dim)")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze backbone weights")
    parser.add_argument("--unfreeze_after", type=int, default=5, help="Unfreeze backbone after this many epochs (0 to disable)")
    args = parser.parse_args()

    set_seed(args.seed)
    cfg: Dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    config_dir = args.config.parent.resolve()
    split_dir_raw = Path(cfg["split_files"]["dir"])
    split_dir = split_dir_raw if split_dir_raw.is_absolute() else (config_dir / split_dir_raw).resolve()

    train_ids = read_split_ids(split_dir, "train")
    val_ids   = read_split_ids(split_dir, "val")
    test_ids = read_split_ids(split_dir, "test") if (split_dir / "test_ids.txt").exists() else None

    # --------- Datasets / Dataloaders ----------
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

    # CXR-friendly aug (no ColorJitter, small rotation)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5, fill=0),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_ds = AugmentedEGDCXRDataset(transform=train_transform, **common_kwargs, case_ids=train_ids)
    val_ds   = AugmentedEGDCXRDataset(transform=val_transform, **common_kwargs, case_ids=val_ids)
    test_ds  = AugmentedEGDCXRDataset(transform=val_transform, **common_kwargs, case_ids=test_ids) if test_ids is not None else None

    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["train"]["num_workers"])
    sampler = None
    if cfg["train"].get("use_weighted_sampler", False):
        w = train_ds.sample_weights()
        sampler = WeightedRandomSampler(w.double(), len(w), replacement=True)

    train_loader = create_dataloader(train_ds, batch_size=bs, shuffle=(sampler is None), sampler=sampler, num_workers=nw)
    val_loader   = create_dataloader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw)
    test_loader  = create_dataloader(test_ds,  batch_size=bs, shuffle=False, num_workers=nw) if test_ds is not None else None

    # --------------- Model ----------------
    yaml_backbone = (cfg.get("options") or {}).get("image_backbone", "resnet50")
    image_backbone = args.image_backbone or yaml_backbone

    model = ImageCNN(
        num_classes=len(cfg["train"]["classes"]),
        image_backbone=image_backbone,
        d_img=int(cfg["model"].get("d_img", 128)),
        use_proj=not args.no_proj,
        freeze_backbone=bool(args.freeze_backbone),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Handle staged unfreezing
    if args.unfreeze_after > 0:
        print(f"Starting with frozen backbone; will unfreeze after epoch {args.unfreeze_after}")
        for p in model.backbone.parameters():
            p.requires_grad = False

    # --------------- Optimizer ------------
    opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))

    # LR scheduler
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=3)

    # ------------- Train / Early stop -----
    epochs   = int(cfg["train"]["epochs"])
    patience = int(cfg["train"].get("patience", 8))
    ckpt_dir = Path(cfg["output_path"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / (cfg["output_path"].get("best_ckpt_name") or "cnn_best.pt")

    best_val_loss = float("inf")
    bad = 0

    for ep in range(1, epochs + 1):
        # Unfreeze backbone if applicable
        if args.unfreeze_after > 0 and ep == args.unfreeze_after + 1:
            print(f"[Epoch {ep:02d}] Unfreezing backbone")
            for p in model.backbone.parameters():
                p.requires_grad = True
            opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=opt.param_groups[0]['lr'], weight_decay=float(cfg["train"]["weight_decay"]))

        tr_loss, tr_acc = run_one_epoch(model, train_loader, cfg, device, optimizer=opt, train=True)

        va_loss, va_acc, va_auc, va_report = evaluate_epoch(model, val_loader, cfg, device)

        # Pretty print metrics
        msg = (f"[Epoch {ep:02d}] train: loss={tr_loss:.4f} acc={tr_acc:.3f} | "
               f"val: loss={va_loss:.4f} acc={va_acc:.3f}")
        if va_auc is not None:
            msg += f" macroAUC={va_auc:.3f}"
        print(msg)
        if va_report is not None:
            print("Per-class (val):\n" + va_report)

        # Step scheduler
        scheduler.step(va_loss)

        # Save best
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            bad = 0
            torch.save({
                "model": model.state_dict(),
                "cfg": cfg,
                "epoch": ep,
                "val_loss": best_val_loss,
                "backbone": image_backbone,
                "use_proj": not args.no_proj,
                "freeze_backbone": bool(args.freeze_backbone),
            }, best_path)
            print(f"  → Saved best checkpoint: {best_path}")
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping (no improvement for {patience} epochs).")
                break

    # optional test pass
    if test_loader is not None:
        te_loss, te_acc, te_auc, te_report = evaluate_epoch(model, test_loader, cfg, device)
        msg = f"[Test] loss={te_loss:.4f} acc={te_acc:.3f}"
        if te_auc is not None:
            msg += f" macroAUC={te_auc:.3f}"
        print(msg)
        if te_report is not None:
            print("Per-class (test):\n" + te_report)

    print("Training finished.")


if __name__ == "__main__":
    main()
