#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, random, time, sys
from pathlib import Path
from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from omegaconf import OmegaConf
from tqdm import tqdm

# Ensure local src/ is importable
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.models.tmrnn import TMRNN
from egd_cxr_dataset.datasets.egd_cxr import EGDCXRDataset, create_dataloader  # type: ignore
from transformers import T5Tokenizer


def encode_batch_transcripts(transcripts: List[Dict[str, Any]], tokenizer: T5Tokenizer, max_len: int, device: torch.device) -> torch.Tensor:
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


def train_one_epoch(model: TMRNN, dl, tok: T5Tokenizer, cfg) -> Tuple[float, float, float]:
    device = next(model.parameters()).device
    model.train()
    ce = nn.CrossEntropyLoss(label_smoothing=float(cfg.train.label_smoothing))
    total_loss = total_cls = total_txt = n = 0.0

    pbar = tqdm(dl, desc="train", leave=False, dynamic_ncols=True)
    for batch in pbar:
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
            elif isinstance(v, dict):
                for sk, sv in v.items():
                    if isinstance(sv, torch.Tensor):
                        batch[k][sk] = sv.to(device)

        txt_ids = encode_batch_transcripts(batch["transcripts"], tok, cfg.train.max_txt_len, device)

        cls_logits, txt_logits = model(batch, transcripts_ids=txt_ids)
        y = batch["labels"]["single_index"].to(device).view(-1)

        loss_cls = ce(cls_logits, y)
        pad_id = tok.pad_token_id
        loss_txt = 0.0
        if txt_ids is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)
            loss_txt = loss_fct(txt_logits.view(-1, txt_logits.size(-1)), txt_ids.view(-1))

        loss = loss_cls + float(cfg.train.lambda_txt) * loss_txt

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_cls += loss_cls.item() * bs
        total_txt += float(loss_txt) * bs
        n += bs

        # live metrics on progress bar
        pbar.set_postfix({
            "loss": f"{(total_loss / max(1,n)):.3f}",
            "cls": f"{(total_cls / max(1,n)):.3f}",
            "txt": f"{(total_txt / max(1,n)):.3f}",
        })

    return total_loss / n, total_cls / n, total_txt / n


def evaluate(model: TMRNN, dl, tok: T5Tokenizer, cfg) -> Tuple[float, float]:
    device = next(model.parameters()).device
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = total_acc = n = 0.0

    with torch.no_grad():
        pbar = tqdm(dl, desc="eval", leave=False, dynamic_ncols=True)
        for batch in pbar:
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
                elif isinstance(v, dict):
                    for sk, sv in v.items():
                        if isinstance(sv, torch.Tensor):
                            batch[k][sk] = sv.to(device)

            cls_logits, _ = model(batch, transcripts_ids=None)
            y = batch["labels"]["single_index"].to(device).view(-1)
            loss = ce(cls_logits, y)
            pred = cls_logits.argmax(dim=1)
            acc = (pred == y).float().mean().item()

            bs = y.size(0)
            total_loss += loss.item() * bs
            total_acc += acc * bs
            n += bs

            pbar.set_postfix({
                "loss": f"{(total_loss / max(1,n)):.3f}",
                "acc": f"{(total_acc / max(1,n)):.3f}",
            })

    return total_loss / n, total_acc / n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    set_seed(42)

    root = Path(cfg.input_path.gaze_raw)
    seg = Path(cfg.input_path.segmentation_dir)
    tr = Path(cfg.input_path.transcripts_dir)
    dicom = Path(cfg.input_path.dicom_raw) if cfg.input_path.get("dicom_raw") else None

    split_dir = Path(cfg.split_files.dir)
    train_ids, val_ids, test_ids = None, None, None
    has_train = (split_dir / "train_ids.txt").exists()
    has_val = (split_dir / "val_ids.txt").exists()
    has_test = (split_dir / "test_ids.txt").exists()
    if has_train and has_val:
        train_ids = [l.strip() for l in (split_dir / "train_ids.txt").read_text().splitlines() if l.strip() and not l.startswith('#')]
        val_ids   = [l.strip() for l in (split_dir / "val_ids.txt").read_text().splitlines()   if l.strip() and not l.startswith('#')]
        if has_test:
            test_ids = [l.strip() for l in (split_dir / "test_ids.txt").read_text().splitlines() if l.strip() and not l.startswith('#')]

    train_ds = EGDCXRDataset(
        root=root, seg_path=seg, transcripts_path=tr, dicom_root=dicom,
        max_fixations=cfg.train.max_fixations,
        case_ids=train_ids, classes=cfg.train.classes,
    )
    if val_ids is None:
        all_ids = train_ds.case_ids
        split = int(0.8 * len(all_ids))
        train_ids, val_ids = all_ids[:split], all_ids[split:]
        val_ds = EGDCXRDataset(root=root, seg_path=seg, transcripts_path=tr, dicom_root=dicom,
                               max_fixations=cfg.train.max_fixations, case_ids=val_ids, classes=cfg.train.classes)
    else:
        val_ds = EGDCXRDataset(root=root, seg_path=seg, transcripts_path=tr, dicom_root=dicom,
                               max_fixations=cfg.train.max_fixations, case_ids=val_ids, classes=cfg.train.classes)

    # test dataset (optional; fallback to val)
    if test_ids is not None:
        test_ds = EGDCXRDataset(root=root, seg_path=seg, transcripts_path=tr, dicom_root=dicom,
                                max_fixations=cfg.train.max_fixations, case_ids=test_ids, classes=cfg.train.classes)
    else:
        test_ds = val_ds

    sampler = build_samplers(train_ds, cfg.train.get("use_weighted_sampler", True))
    train_dl = create_dataloader(train_ds, batch_size=cfg.train.batch_size, shuffle=(sampler is None), sampler=sampler, num_workers=cfg.train.num_workers)
    val_dl = create_dataloader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, sampler=None, num_workers=cfg.train.num_workers)
    test_dl = create_dataloader(test_ds, batch_size=cfg.train.batch_size, shuffle=False, sampler=None, num_workers=cfg.train.num_workers)

    model = TMRNN(
        num_seg=(train_ds.num_segments or 0) if hasattr(train_ds, 'num_segments') else len(getattr(train_ds, 'region_names', [])),
        num_box=train_ds.num_box_classes,
        num_classes=len(train_ds.class_names) if hasattr(train_ds, 'class_names') else len(cfg.train.classes),
        d_g=cfg.model.d_g, d_r=cfg.model.d_s, d_img=cfg.model.d_img, d_h=cfg.model.d_h,
        t5_name="google-t5/t5-base", freeze_t5=True, use_attn_pool=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    global optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    tok = model.tok

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

    total_start = time.time()
    for epoch in range(int(cfg.train.epochs)):
        ep_start = time.time()
        tr_loss, tr_cls, tr_txt = train_one_epoch(model, train_dl, tok, cfg)
        val_loss, val_acc = evaluate(model, val_dl, tok, cfg)
        ep_time = time.time() - ep_start
        print(f"Epoch {epoch+1:03d} | time={ep_time:.2f}s | train: loss={tr_loss:.4f} cls={tr_cls:.4f} txt={tr_txt:.4f} | val: loss={val_loss:.4f} acc={val_acc:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            bad = 0
            torch.save({"model": model.state_dict(), "cfg": OmegaConf.to_container(cfg, resolve=True)}, best_path)
            print(f"  ↳ saved best checkpoint to: {best_path}")
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    total_time = time.time() - total_start
    test_loss, test_acc = evaluate(model, test_dl, tok, cfg)
    print(f"\nTraining done in {total_time/60:.2f} min. Test: loss={test_loss:.4f} acc={test_acc:.4f}")
    print("Done.")


