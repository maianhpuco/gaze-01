#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys, yaml, os, math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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

from egd_cxr_dataset.datasets.egd_cxr import EGDCXRDataset, create_dataloader  # type: ignore
from src.models.tmrnn import TMRNN, build_vocabulary

# Optional AUC metrics
try:
    from sklearn.metrics import roc_auc_score
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
        if s and not s.startswith("#"):
            ids.append(s)
    if not ids:
        raise FileNotFoundError(f"[ERROR] Split file exists but is empty: {path}")
    return ids

def _move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            for sk, sv in v.items():
                if isinstance(sv, torch.Tensor):
                    batch[k][sk] = sv.to(device, non_blocking=True)
    return batch

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    correct = (pred == targets).float().sum().item()
    return correct / max(1, targets.numel())

def encode_batch_transcripts(transcripts, tokenizer, max_len: int, device: torch.device) -> torch.Tensor:
    """Pad-only when empty, else standard encode (T5 tokenizer)."""
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


# --------- Train / Val loops ----------
def run_one_epoch(
    model: TMRNN,
    loader,
    cfg: Dict[str, Any],
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    train: bool = True,
) -> Tuple[float, float, float, float]:
    """
    Returns (avg_total_loss, avg_cls_loss, avg_txt_loss, acc)
    """
    model.train(train)
    ce_cls = nn.CrossEntropyLoss(label_smoothing=float(cfg["train"].get("label_smoothing", 0.0))) if train else nn.CrossEntropyLoss()

    total_loss = total_cls = total_txt = 0.0
    total_correct = 0
    total_seen = 0

    it = tqdm(loader, disable=not train, leave=False)
    for batch in it:
        batch = _move_to_device(batch, device)

        # Teacher forcing only when training and text head enabled
        transcripts_ids = None
        if train and model.enable_text_head and model.use_teacher_forcing and (model.tok is not None):
            transcripts_ids = encode_batch_transcripts(
                batch["transcripts"], model.tok, int(cfg["train"]["max_txt_len"]), device
            )

        cls_logits, txt_logits = model(batch, transcripts_ids=transcripts_ids)
        y = batch["labels"]["single_index"].view(-1)

        loss_cls = ce_cls(cls_logits, y)
        loss_txt = torch.tensor(0.0, device=device)
        if train and txt_logits is not None and transcripts_ids is not None:
            pad_id = model.tok.pad_token_id
            loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)
            loss_txt = loss_fct(
                txt_logits.reshape(-1, txt_logits.size(-1)),
                transcripts_ids.reshape(-1)
            )
        loss = loss_cls + float(cfg["train"].get("lambda_txt", 1.0)) * loss_txt

        if train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += float(loss)
        total_cls  += float(loss_cls)
        total_txt  += float(loss_txt)

        with torch.no_grad():
            preds = cls_logits.argmax(dim=1)
            total_correct += (preds == y).float().sum().item()
            total_seen    += y.numel()

    denom = max(1, len(loader))
    acc = total_correct / max(1, total_seen)
    return total_loss / denom, total_cls / denom, total_txt / denom, acc


@torch.no_grad()
def evaluate_epoch(
    model: TMRNN,
    loader,
    device: torch.device,
    tta_hflip: bool = True,
) -> Tuple[float, float, Optional[float], Optional[np.ndarray]]:
    """
    Returns: (avg_loss, avg_acc, macro_auc (or None), per_class_auc (or None))
    """
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss, total_acc, n = 0.0, 0.0, 0

    compute_auc = HAS_SK
    if compute_auc:
        all_probs, all_y = [], []

    for batch in loader:
        batch = _move_to_device(batch, device)
        logits, _ = model(batch)

        if tta_hflip:
            # Only flip images; keep sequence streams intact
            batch_tta = dict(batch)
            batch_tta["images"] = batch["images"].flip(dims=[-1])  # HFlip width dim
            logits_tta, _ = model(batch_tta)
            logits = (logits + logits_tta) * 0.5

        y = batch["labels"]["single_index"].view(-1)
        loss = ce(logits, y)
        acc = accuracy(logits, y)

        bsz = y.size(0)
        total_loss += loss.item() * bsz
        total_acc  += acc * bsz
        n += bsz

        if compute_auc:
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            all_probs.append(probs)
            all_y.append(y.detach().cpu().numpy())

    macro_auc, per_class_auc = None, None
    if compute_auc and n > 0:
        probs = np.concatenate(all_probs, axis=0)
        y = np.concatenate(all_y, axis=0)
        try:
            macro_auc = roc_auc_score(y, probs, multi_class="ovr")
            C = probs.shape[1]
            y_1hot = np.eye(C)[y]
            aucs = []
            for c in range(C):
                if y_1hot[:, c].sum() > 0 and y_1hot[:, c].sum() < len(y):
                    aucs.append(roc_auc_score(y_1hot[:, c], probs[:, c]))
                else:
                    aucs.append(np.nan)
            per_class_auc = np.array(aucs)
        except Exception:
            macro_auc, per_class_auc = None, None

    return total_loss / max(1, n), total_acc / max(1, n), macro_auc, per_class_auc


# ------------------ Main -----------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=2025)

    # parity with CNN switches
    ap.add_argument("--image_backbone", type=str, default=None,
                    choices=["resnet18", "resnet50", "densenet121", "txrv_densenet121"])
    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--unfreeze_after", type=int, default=5, help="Unfreeze backbone after N epochs (0=disabled)")

    # modality toggles (optional)
    ap.add_argument("--no_gaze", action="store_true")
    ap.add_argument("--no_roi", action="store_true")
    ap.add_argument("--no_text", action="store_true", help="disable teacher-forcing")
    ap.add_argument("--no_text_decode", action="store_true", help="disable the text head entirely")

    args = ap.parse_args()
    set_seed(args.seed)

    cfg: Dict[str, Any] = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve split dir robustly
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
    test_ids  = read_split_ids(split_dir, "test") if (split_dir / "test_ids.txt").exists() else None

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
    test_ds  = EGDCXRDataset(**common_kwargs, case_ids=test_ids) if test_ids is not None else None

    bs = int(cfg["train"]["batch_size"]); nw = int(cfg["train"]["num_workers"])
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

    model = TMRNN(
        num_seg=getattr(train_ds, "num_segments", None) or len(train_ds.region_names),
        num_box=train_ds.num_box_classes,
        d_g=cfg["model"]["d_g"],
        d_r=cfg["model"]["d_s"],
        d_img=cfg["model"]["d_img"],
        d_h=cfg["model"]["d_h"],
        num_classes=len(cfg["train"]["classes"]),
        image_backbone=image_backbone,
        use_gaze=not args.no_gaze,
        use_roi=not args.no_roi,
        enable_text_head=not args.no_text_decode,
        use_teacher_forcing=not args.no_text,
    ).to(device)

    # Optional freeze at start
    if args.freeze_backbone:
        for p in model.img_encoder.parameters():
            p.requires_grad = False

    # Vocab meta (for completeness if text head on)
    if model.enable_text_head:
        (Path(cfg["output_path"]["checkpoint_dir"]) / "meta").mkdir(parents=True, exist_ok=True)
        with (Path(cfg["output_path"]["checkpoint_dir"]) / "meta" / "tok.txt").open("w") as f:
            f.write(model.tok.name_or_path if model.tok is not None else "none")

    # ---------- Optimizer (2 groups: smaller LR for backbone) ----------
    base_lr = float(cfg["train"]["lr"])
    wd = float(cfg["train"]["weight_decay"])
    backbone_lr_mult = float(cfg["train"].get("backbone_lr_mult", 0.1))

    head_params = list(model.img_proj.parameters()) + list(model.encoder.parameters()) + \
                  list(model.cls_head.parameters())
    if model.use_gaze:
        head_params += list(model.gaze_mlp.parameters())
    if model.use_roi:
        head_params += list(model.roi_proj.parameters())
    if model.use_attn_pool:
        head_params += list(model.attn_pool.parameters())
    if model.enable_text_head and model.h_to_t5 is not None:
        head_params += list(model.h_to_t5.parameters())

    backbone_params = [p for p in model.img_encoder.parameters() if p.requires_grad]

    opt = AdamW(
        [
            {"params": backbone_params, "lr": base_lr * backbone_lr_mult},
            {"params": head_params,     "lr": base_lr},
        ],
        weight_decay=wd
    )

    # ---------- OneCycleLR (per-batch) ----------
    from torch.optim.lr_scheduler import OneCycleLR
    epochs   = int(cfg["train"]["epochs"])
    steps_per_epoch = max(1, math.ceil(len(train_loader)))
    scheduler = OneCycleLR(
        opt, max_lr=[base_lr * backbone_lr_mult, base_lr],
        epochs=epochs, steps_per_epoch=steps_per_epoch,
        pct_start=0.1, div_factor=10.0, final_div_factor=100.0
    )

    # ------------- Train / Early stop (by macro-AUC) -----
    patience = int(cfg["train"].get("patience", 10))
    tta_hflip = bool(cfg["train"].get("tta_hflip", True))
    ckpt_dir = Path(cfg["output_path"]["checkpoint_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / (cfg["output_path"].get("best_ckpt_name") or "tmrnn_best.pt")

    best_key = -float("inf")  # macroAUC
    bad = 0

    for ep in range(1, epochs + 1):
        # staged unfreeze
        if args.unfreeze_after > 0 and ep == args.unfreeze_after + 1:
            for p in model.img_encoder.parameters():
                p.requires_grad = True
            # rebuild optimizer w/ new requires_grad
            backbone_params = [p for p in model.img_encoder.parameters() if p.requires_grad]
            opt = AdamW(
                [
                    {"params": backbone_params, "lr": base_lr * backbone_lr_mult},
                    {"params": head_params,     "lr": base_lr},
                ],
                weight_decay=wd
            )
            scheduler = OneCycleLR(
                opt, max_lr=[base_lr * backbone_lr_mult, base_lr],
                epochs=epochs, steps_per_epoch=steps_per_epoch,
                pct_start=0.1, div_factor=10.0, final_div_factor=100.0
            )
            print(f"[Epoch {ep:02d}] Unfroze backbone")

        # train (scheduler per batch)
        tr_loss, tr_cls, tr_txt, tr_acc = run_one_epoch(model, train_loader, cfg, device, optimizer=opt, train=True)
        # manually step OneCycleLR per batch already; nothing here

        # val
        va_loss, va_acc, va_auc, va_aucs = evaluate_epoch(model, val_loader, device, tta_hflip=tta_hflip)

        # ----- Clean, 2-line logging -----
        print(f"[Epoch {ep:02d}] TRAIN  loss={tr_loss:.4f}  acc={tr_acc:.3f}  (cls={tr_cls:.4f}, txt={tr_txt:.4f})")
        if va_auc is not None and va_aucs is not None:
            va_aucs_str = ",".join([f"{x:.3f}" if not np.isnan(x) else "nan" for x in va_aucs.tolist()])
            print(f"[Epoch {ep:02d}] VAL    loss={va_loss:.4f}  acc={va_acc:.3f}  macroAUC={va_auc:.3f}  perClassAUC=[{va_aucs_str}]")
        else:
            print(f"[Epoch {ep:02d}] VAL    loss={va_loss:.4f}  acc={va_acc:.3f}")

        # ----- Save best by macro-AUC (fallback to loss if AUC None) -----
        current_key = va_auc if (va_auc is not None) else (-va_loss)
        if current_key > best_key:
            best_key = current_key
            bad = 0
            torch.save({
                "model": model.state_dict(),
                "cfg": cfg,
                "epoch": ep,
                "val_metric": float(best_key),
                "backbone": image_backbone,
            }, best_path)
            print(f"  → Saved best checkpoint (by macroAUC): {best_path}")
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping (no macroAUC improvement for {patience} epochs).")
                break

    # ----- TEST on best checkpoint -----
    if test_loader is not None and best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        te_loss, te_acc, te_auc, te_aucs = evaluate_epoch(model, test_loader, device, tta_hflip=tta_hflip)
        if te_auc is not None and te_aucs is not None:
            te_aucs_str = ",".join([f"{x:.3f}" if not np.isnan(x) else "nan" for x in te_aucs.tolist()])
            print(f"[Test] LOSS={te_loss:.4f}  ACC={te_acc:.3f}  macroAUC={te_auc:.3f}  perClassAUC=[{te_aucs_str}]")
        else:
            print(f"[Test] LOSS={te_loss:.4f}  ACC={te_acc:.3f}")
    else:
        print("[Test] Skipped (no test split or no best checkpoint).")

    print("Training finished.")


if __name__ == "__main__":
    main()
