# ──────────────────────────────────────────────────────────────────────────────
# File: main_train.py
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import argparse, os, random, time
from pathlib import Path
from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from omegaconf import OmegaConf
import sys 

# local imports
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC)) 
    
from src.models.tmrnn_gold_gr import TMRNN
from egd_cxr_dataset.datasets.egd_cxr import EGDCXRDataset, create_dataloader
from transformers import T5Tokenizer


# ----------------------------
# transcript utilities
# ----------------------------

def encode_batch_transcripts(transcripts: List[Dict[str, Any]], tokenizer: T5Tokenizer, max_len: int, device: torch.device) -> torch.Tensor:
    """Tokenize raw transcript strings. If empty, return all PAD; HF will ignore via label smoothing/ignore index.
    We do not use segment‑level τ here; can be added if desired.
    """
    ids = []
    for tr in transcripts:
        text = (tr.get("text") or "").strip()
        if not text:
            ids.append(torch.full((max_len,), tokenizer.pad_token_id, dtype=torch.long))
        else:
            enc = tokenizer(text, max_length=max_len, truncation=True, padding="max_length", return_tensors="pt")
            ids.append(enc.input_ids.squeeze(0))
    return torch.stack(ids, dim=0).to(device)


def set_seed(s: int):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def maybe_make_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def build_samplers(dataset: EGDCXRDataset, use_weighted: bool) -> WeightedRandomSampler | None:
    if not use_weighted:
        return None
    weights = dataset.sample_weights().double()
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


# ----------------------------
# training / eval loops
# ----------------------------

def train_one_epoch(model: TMRNN, dl, tok: T5Tokenizer, cfg) -> Tuple[float, float, float]:
    device = next(model.parameters()).device
    model.train()

    ce = nn.CrossEntropyLoss(label_smoothing=float(cfg.train.label_smoothing))
    total_loss = total_cls = total_txt = n = 0.0

    for batch in dl:
        # move tensors
        for k,v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
            elif isinstance(v, dict):
                for sk, sv in v.items():
                    if isinstance(sv, torch.Tensor):
                        batch[k][sk] = sv.to(device)

        # encode transcripts
        txt_ids = encode_batch_transcripts(batch["transcripts"], tok, cfg.train.max_txt_len, device)

        # forward
        cls_logits, txt_logits = model(batch, transcripts_ids=txt_ids)

        # losses
        y = batch["labels"]["single_index"].to(device).view(-1)
        loss_cls = ce(cls_logits, y)

        # Use HF loss by re‑calling T5 with labels to avoid manual shifting (cheap):
        # (We can reuse encoder states via model.forward, but fine for clarity.)
        loss_txt = 0.0
        if txt_ids is not None:
            # Recompute txt loss with HF’s internal label shifting
            with torch.no_grad():
                pass  # logits already computed; loss stability comes from CE below if you prefer
            # Manual CE on logits from forward (ignore PAD)
            pad_id = tok.pad_token_id
            txt_tgt = txt_ids
            logits = txt_logits
            loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)
            loss_txt = loss_fct(logits.view(-1, logits.size(-1)), txt_tgt.view(-1))

        loss = loss_cls + float(cfg.train.lambda_txt) * loss_txt

        # step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # meters
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_cls  += loss_cls.item() * bs
        total_txt  += float(loss_txt) * bs
        n += bs

    return total_loss / n, total_cls / n, total_txt / n


def evaluate(model: TMRNN, dl, tok: T5Tokenizer, cfg) -> Tuple[float, float]:
    device = next(model.parameters()).device
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = total_acc = n = 0.0

    with torch.no_grad():
        for batch in dl:
            for k,v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
                elif isinstance(v, dict):
                    for sk, sv in v.items():
                        if isinstance(sv, torch.Tensor):
                            batch[k][sk] = sv.to(device)

            # dummy transcripts to enable shape; not used for val loss here
            txt_ids = encode_batch_transcripts(batch["transcripts"], tok, cfg.train.max_txt_len, device)
            cls_logits, _ = model(batch, transcripts_ids=None)

            y = batch["labels"]["single_index"].to(device).view(-1)
            loss = ce(cls_logits, y)

            pred = cls_logits.argmax(dim=1)
            acc = (pred == y).float().mean().item()

            bs = y.size(0)
            total_loss += loss.item() * bs
            total_acc  += acc * bs
            n += bs

    return total_loss / n, total_acc / n


# ----------------------------
# main
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    set_seed(42)

    # paths
    root = Path(cfg.input_path.gaze_raw)
    seg  = Path(cfg.input_path.segmentation_dir)
    tr   = Path(cfg.input_path.transcripts_dir)
    dicom= Path(cfg.input_path.dicom_raw) if cfg.input_path.get("dicom_raw") else None

    # (Optional) split files (train/val) by IDs
    split_dir = Path(cfg.split_files.dir)
    train_ids, val_ids = None, None
    if (split_dir / "train_ids.txt").exists() and (split_dir / "val_ids.txt").exists():
        train_ids = [l.strip() for l in (split_dir/"train_ids.txt").read_text().splitlines() if l.strip() and not l.startswith('#')]
        val_ids   = [l.strip() for l in (split_dir/"val_ids.txt").read_text().splitlines() if l.strip() and not l.startswith('#')]

    # datasets
    train_ds = EGDCXRDataset(
        root=root, seg_path=seg, transcripts_path=tr, dicom_root=dicom,
        max_fixations=cfg.train.max_fixations,
        case_ids=train_ids, classes=cfg.train.classes,
    )
    if val_ids is None:
        # simple split if no provided list (80/20)
        all_ids = train_ds.case_ids
        split = int(0.8 * len(all_ids))
        train_ids, val_ids = all_ids[:split], all_ids[split:]
        val_ds = EGDCXRDataset(root=root, seg_path=seg, transcripts_path=tr, dicom_root=dicom,
                               max_fixations=cfg.train.max_fixations, case_ids=val_ids, classes=cfg.train.classes)
    else:
        val_ds = EGDCXRDataset(root=root, seg_path=seg, transcripts_path=tr, dicom_root=dicom,
                               max_fixations=cfg.train.max_fixations, case_ids=val_ids, classes=cfg.train.classes)

    # samplers / loaders
    sampler = build_samplers(train_ds, cfg.train.get("use_weighted_sampler", True))
    train_dl = create_dataloader(train_ds, batch_size=cfg.train.batch_size, shuffle=(sampler is None), sampler=sampler, num_workers=cfg.train.num_workers)
    val_dl   = create_dataloader(val_ds,   batch_size=cfg.train.batch_size, shuffle=False, sampler=None, num_workers=cfg.train.num_workers)

    # model
    model = TMRNN(
        num_seg=(train_ds.num_segments or 0),
        num_box=train_ds.num_box_classes,
        num_classes=len(train_ds.class_names),
        d_g=cfg.model.d_g, d_r=cfg.model.d_s, d_img=cfg.model.d_img, d_h=cfg.model.d_h,
        t5_name="google-t5/t5-base", freeze_t5=True, use_attn_pool=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # optimizer / scheduler
    global optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    tok = model.tok

    # ckpt paths
    ckpt_dir = Path(cfg.output_path.checkpoint_dir)
    maybe_make_dir(ckpt_dir)
    best_name = cfg.output_path.get("best_ckpt_name", None)
    if not best_name:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        best_name = f"tmrnn_{cfg.dataset_name}_{stamp}_best.pt"
    best_path = ckpt_dir / best_name

    best_val = float("inf")
    patience = int(cfg.train.patience)
    bad = 0

    for epoch in range(int(cfg.train.epochs)):
        tr_loss, tr_cls, tr_txt = train_one_epoch(model, train_dl, tok, cfg)
        val_loss, val_acc = evaluate(model, val_dl, tok, cfg)

        print(f"Epoch {epoch+1:03d} | train: loss={tr_loss:.4f} cls={tr_cls:.4f} txt={tr_txt:.4f} | val: loss={val_loss:.4f} acc={val_acc:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            bad = 0
            torch.save({
                "model": model.state_dict(),
                "cfg": OmegaConf.to_container(cfg, resolve=True),
            }, best_path)
            print(f"  ↳ saved best checkpoint to: {best_path}")
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    print("Done.")

