# inference_classify.py
from __future__ import annotations
import sys, yaml, json
from pathlib import Path
from typing import Dict, Any, List
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

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

@torch.no_grad()
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    # Optional overrides (keep consistent with training)
    ap.add_argument("--image_backbone", type=str, default=None,
                    choices=["resnet18","resnet50","densenet121","txrv_densenet121"])
    ap.add_argument("--no_gaze", action="store_true")
    ap.add_argument("--no_roi", action="store_true")
    ap.add_argument("--no_text_decode", action="store_true")
    ap.add_argument("--no_text", action="store_true")
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
    loader = create_dataloader(ds, batch_size=64, shuffle=False, num_workers=2)

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

    y_true, y_pred, y_prob = [], [], []

    for batch in loader:
        # move
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
            elif isinstance(v, dict):
                for sk, sv in v.items():
                    if isinstance(sv, torch.Tensor):
                        batch[k][sk] = sv.to(device)
        logits, _ = model(batch, transcripts_ids=None)
        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(1).cpu().tolist()
        y_pred.extend(pred)
        y_prob.extend(probs.cpu().tolist())
        y_true.extend(batch["labels"]["single_index"].cpu().tolist())

    print(classification_report(y_true, y_pred, target_names=cfg["train"]["classes"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    out_dir = Path(cfg["output_path"]["checkpoint_dir"]) / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "test_probs.json").open("w") as f:
        json.dump(dict(y_true=y_true, y_pred=y_pred, y_prob=y_prob), f)
    print(f"Saved probabilities to {out_dir / 'test_probs.json'}")

if __name__ == "__main__":
    main()
