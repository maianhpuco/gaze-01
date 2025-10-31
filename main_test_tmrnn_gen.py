# inference_generate.py
from __future__ import annotations
import sys, yaml, json, math
from pathlib import Path
from typing import Dict, Any, List, Tuple
import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from egd_cxr_dataset.datasets.egd_cxr import EGDCXRDataset, create_dataloader
from src.models.tmrnn import TMRNN


def read_split_ids(split_dir: Path, split: str) -> List[str]:
    p = split_dir / f"{split}_ids.txt"
    return [l.strip() for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")]

def resolve_split_dir(args_cfg_path: Path, cfg: Dict[str, Any]) -> Path:
    project_root = Path(__file__).resolve().parent
    config_dir = args_cfg_path.parent.resolve()
    split_dir_raw = Path(cfg["split_files"]["dir"])
    if split_dir_raw.is_absolute():
        return split_dir_raw
    first = split_dir_raw.parts[0] if split_dir_raw.parts else ""
    if first.startswith("configs"):
        return (project_root / split_dir_raw).resolve()
    return (config_dir / split_dir_raw).resolve()


# ----------------- Simple metrics (no external deps) -----------------

def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)] if len(tokens) >= n else []

def bleu_1_2(ref: str, hyp: str) -> Dict[str, float]:
    # BLEU-1 and BLEU-2 with brevity penalty
    ref_toks = ref.lower().split()
    hyp_toks = hyp.lower().split()
    if not hyp_toks:
        return {"bleu1": 0.0, "bleu2": 0.0}

    def precision(n):
        ref_ngr = _ngrams(ref_toks, n); hyp_ngr = _ngrams(hyp_toks, n)
        if not hyp_ngr:
            return 0.0
        ref_counts = {}
        for g in ref_ngr:
            ref_counts[g] = ref_counts.get(g, 0) + 1
        match = 0
        used = {}
        for g in hyp_ngr:
            cnt = used.get(g, 0)
            if ref_counts.get(g, 0) > cnt:
                used[g] = cnt + 1
                match += 1
        return match / max(1, len(hyp_ngr))

    p1 = precision(1)
    p2 = precision(2)

    # brevity penalty
    r = len(ref_toks); c = len(hyp_toks)
    bp = 1.0 if c > r else math.exp(1 - r / max(1, c))

    bleu1 = bp * p1
    bleu2 = bp * math.sqrt(p1 * p2) if p1 > 0 and p2 > 0 else 0.0
    return {"bleu1": float(bleu1), "bleu2": float(bleu2)}

def rouge_l(ref: str, hyp: str) -> float:
    # LCS-based F-measure
    a = ref.lower().split(); b = hyp.lower().split()
    if not a or not b:
        return 0.0
    # LCS DP
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[-1][-1]
    prec = lcs / len(b); rec = lcs / len(a)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


@torch.no_grad()
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--max_len", type=int, default=64)
    # Switches to mirror the trained config if needed
    ap.add_argument("--image_backbone", type=str, default=None,
                    choices=["resnet18","resnet50","densenet121","txrv_densenet121"])
    ap.add_argument("--no_gaze", action="store_true")
    ap.add_argument("--no_roi", action="store_true")
    ap.add_argument("--no_text_decode", action="store_true")   # if set, transcript gen will be disabled
    ap.add_argument("--no_text", action="store_true")          # teacher forcing (irrelevant at inference, safe)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_dir = resolve_split_dir(args.config, cfg)
    test_ids = read_split_ids(split_dir, "test")

    in_paths = cfg["input_path"]
    ds = EGDCXRDataset(
        root=Path(in_paths["gaze_raw"]),
        seg_path=Path(in_paths["segmentation_dir"]),
        transcripts_path=Path(in_paths["transcripts_dir"]),
        dicom_root=Path(in_paths["dicom_raw"]),
        max_fixations=cfg["train"]["max_fixations"],
        classes=cfg["train"]["classes"],
        case_ids=test_ids,
        drop_unlabelled=True,
    )
    loader = create_dataloader(ds, batch_size=16, shuffle=False, num_workers=2)

    opts = cfg.get("options", {})
    def opt(key, default):
        v_cli = getattr(args, key, None)
        return v_cli if v_cli is not None else opts.get(key, default)

    model = TMRNN(
        num_seg=getattr(ds, "num_segments", None) or len(ds.region_names),
        num_box=ds.num_box_classes,
        d_g=cfg["model"]["d_g"],
        d_r=cfg["model"]["d_s"],
        d_img=cfg["model"]["d_img"],
        d_h=cfg["model"]["d_h"],
        num_classes=len(cfg["train"]["classes"]),
        image_backbone=opt("image_backbone", "resnet50"),
        use_gaze=not args.no_gaze if hasattr(args,"no_gaze") else True,
        use_roi=not args.no_roi if hasattr(args,"no_roi") else True,
        enable_text_head=not args.no_text_decode if hasattr(args,"no_text_decode") else True,
        use_teacher_forcing=not args.no_text if hasattr(args,"no_text") else True,
    ).to(device)

    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    out_dir = Path(cfg["output_path"]["checkpoint_dir"]) / "inference_transcripts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "test_transcripts.jsonl"

    agg_bleu1 = agg_bleu2 = agg_rougel = n = 0

    with out_jsonl.open("w", encoding="utf-8") as f:
        for batch in loader:
            # move relevant tensors
            batch["images"] = batch["images"].to(device)
            for k, v in batch["fixations"].items():
                if isinstance(v, torch.Tensor):
                    batch["fixations"][k] = v.to(device)

            hyps = model.generate_transcript(batch, max_len=args.max_len) \
                   if model.enable_text_head else [""] * batch["images"].size(0)

            # Ground truth text & ID
            B = batch["images"].size(0)
            for i in range(B):
                # best-effort IDs (dataset may store it differently)
                dicom_id = None
                try:
                    dicom_id = batch.get("meta", {}).get("dicom_id", [None]*B)[i]
                except Exception:
                    pass
                if dicom_id is None:
                    dicom_id = batch.get("case_ids", [None]*B)[i]
                if dicom_id is None:
                    dicom_id = f"case_{n+i:06d}"

                ref = ""
                try:
                    ref = batch["transcripts"][i]["text"]
                except Exception:
                    ref = ""

                hyp = hyps[i]
                m = bleu_1_2(ref, hyp)
                r = rouge_l(ref, hyp)

                agg_bleu1 += m["bleu1"]; agg_bleu2 += m["bleu2"]; agg_rougel += r; n += 1

                f.write(json.dumps({
                    "dicom_id": dicom_id,
                    "ref": ref,
                    "hyp": hyp,
                    "bleu1": m["bleu1"],
                    "bleu2": m["bleu2"],
                    "rougeL": r
                }) + "\n")

    if n > 0:
        print(f"BLEU-1: {agg_bleu1/n:.4f} | BLEU-2: {agg_bleu2/n:.4f} | ROUGE-L: {agg_rougel/n:.4f}")
        print(f"Saved to {out_jsonl}")
    else:
        print("No samples to evaluate.")

if __name__ == "__main__":
    main()
