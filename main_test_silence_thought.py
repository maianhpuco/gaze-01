#!/usr/bin/env python3
"""
Evaluate a trained silence-thought checkpoint on the EGD-CXR test split.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

try:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    SmoothingFunction = None  # type: ignore[assignment]
    sentence_bleu = None  # type: ignore[assignment]

try:
    from rouge_score import rouge_scorer as rouge_lib

    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    rouge_lib = None  # type: ignore[assignment]

# Ensure project sources are importable
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from egd_cxr_dataset import ConfigLoader, EGDCXRDataset, create_dataloader  # noqa: E402
from egd_cxr_dataset.models.gaze_intent_seq_rnn import (  # noqa: E402
    GazeSeqRNNAttend as GazeIntent2TranscriptAndLabels,
)
from egd_cxr_dataset.utils.vocab import Vocab  # noqa: E402
from main_train_silence_thought import (  # noqa: E402
    format_accuracy,
    format_summary,
    read_split_ids,
    run_epoch,
    set_seed,
)


def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _edit_distance(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[n]


def wer(ref: str, hyp: str) -> float:
    ref = _normalize(ref)
    hyp = _normalize(hyp)
    r = ref.split()
    h = hyp.split()
    return 0.0 if len(r) == 0 and len(h) == 0 else (_edit_distance(r, h) / max(1, len(r)))


def cer(ref: str, hyp: str) -> float:
    ref = _normalize(ref)
    hyp = _normalize(hyp)
    return 0.0 if len(ref) == 0 and len(hyp) == 0 else (_edit_distance(list(ref), list(hyp)) / max(1, len(ref)))


def vocab_from_state(state: Dict[str, Any]) -> Vocab:
    """Reconstruct a Vocab object from checkpoint state."""
    itos: Optional[List[str]] = state.get("itos")
    if not itos:
        raise ValueError("Checkpoint is missing vocabulary tokens under key 'itos'")
    stoi = {tok: idx for idx, tok in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)


def build_model_from_sample(
    sample: Dict[str, Any],
    *,
    vocab: Vocab,
    device: torch.device,
    txt_dim: int,
    enc_dim: int,
    use_bbox: bool,
    use_seg: bool,
    use_image: bool,
    use_text: bool,
) -> GazeIntent2TranscriptAndLabels:
    """Instantiate model using metadata inferred from a dataset sample."""
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
    ).to(device)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate silence-thought checkpoint on test data")
    parser.add_argument("--config", type=Path, required=True, help="Dataset configuration YAML")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint to load")
    parser.add_argument("--batch-size", type=int, default=None, help="Override evaluation batch size")
    parser.add_argument("--max-fixations", type=int, default=None, help="Override max fixations")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--max-decode-len", type=int, default=None, help="Override max decode length for samples")
    args = parser.parse_args()

    # Resolve paths and configuration
    config_loader = ConfigLoader(args.config)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hparams: Dict[str, Any] = checkpoint.get("hparams", {})

    batch_size = args.batch_size if args.batch_size is not None else int(hparams.get("batch_size", 4))
    max_fixations = args.max_fixations if args.max_fixations is not None else hparams.get("max_fixations")
    num_workers = args.num_workers if args.num_workers is not None else int(hparams.get("num_workers", 0))
    decode_len = args.max_decode_len if args.max_decode_len is not None else int(hparams.get("max_decode_len", 64))
    seed_value = args.seed if args.seed is not None else int(hparams.get("seed", 0))

    set_seed(seed_value)

    # Prepare paths from configuration
    gaze_root = Path(config_loader.get("input_path", "gaze_raw"))
    seg_dir = Path(config_loader.get("input_path", "segmentation_dir"))
    transcripts_dir = Path(config_loader.get("input_path", "transcripts_dir", default=seg_dir))
    dicom_root = Path(config_loader.get("input_path", "dicom_raw"))

    split_dir_cfg = config_loader.get("split_files", "dir", default=ROOT / "config" / "splits")
    split_dir = Path(split_dir_cfg)
    if not split_dir.is_absolute():
        split_dir = ROOT / split_dir

    test_ids = read_split_ids(split_dir, "test")
    if not test_ids:
        raise ValueError("No test IDs found; cannot evaluate checkpoint")

    test_dataset = EGDCXRDataset(
        root=gaze_root,
        seg_path=seg_dir,
        transcripts_path=transcripts_dir,
        dicom_root=dicom_root,
        max_fixations=max_fixations,
        case_ids=test_ids,
    )

    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty; nothing to evaluate")

    test_loader = create_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    vocab = vocab_from_state(checkpoint["vocab"])
    txt_dim = int(hparams.get("txt_dim", 256))
    enc_dim = int(hparams.get("enc_dim", 256))
    use_bbox = bool(hparams.get("use_bbox", True))
    use_seg = bool(hparams.get("use_seg", True))
    use_image = bool(hparams.get("use_image", True))
    use_text = bool(hparams.get("use_text", True))
    sample = test_dataset[0]
    model = build_model_from_sample(
        sample,
        vocab=vocab,
        device=device,
        txt_dim=txt_dim,
        enc_dim=enc_dim,
        use_bbox=use_bbox,
        use_seg=use_seg,
        use_image=use_image,
        use_text=use_text,
    )
    model.load_state_dict(checkpoint["model_state"])

    label_names = checkpoint.get("label_names", test_dataset.label_proc.schema.class_columns)

    print(f"Evaluating checkpoint {checkpoint_path} on {len(test_ids)} test cases")
    test_loss, test_acc, batches, test_summary = run_epoch(
        model,
        test_loader,
        device=device,
        vocab=vocab,
        optimiser=None,
        desc="test",
    )

    print(f"Test loss {test_loss:.4f} over {batches} batches")
    print("Test summary:   " + format_summary(test_summary))
    print("Test accuracy per class:")
    print("  " + format_accuracy(label_names, test_acc))

    if model.decoder is None:
        print("\nTranscript decoder disabled; skipping text metrics.")
    else:
        print("\nEvaluating transcript metrics (WER/CER/PPL/BLEU/ROUGE/Exact Match)...")
        model.eval()
        text_wer_sum = 0.0
        text_cer_sum = 0.0
        tok_nll_sum = 0.0
        tok_count = 0
        n_cases = 0
        exact_match_total = 0
        bleu_sum = 0.0
        bleu_count = 0
        rouge_sums: Dict[str, float] = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        rouge_count = 0
        bleu_smoothing = SmoothingFunction().method1 if HAS_NLTK else None  # type: ignore[operator]
        rouge_scorer = (
            rouge_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True) if HAS_ROUGE else None  # type: ignore[union-attr]
        )

        with torch.no_grad():
            for batch in test_loader:
                B = batch["labels"]["binary"].size(0)
                for i in range(B):
                    length = int(batch["fixations"]["lengths"][i].item())
                    case = {
                        "xy": batch["fixations"]["xy"][i, :length].to(device),
                        "dwell": batch["fixations"]["dwell"][i, :length].to(device),
                        "time": batch["fixations"]["time"][i, :length].to(device),
                        "seg_hits": batch["fixations"]["seg_hits"][i, :length].to(device),
                        "box_hits": batch["fixations"]["box_hits"][i, :length].to(device),
                    }
                    img = batch["images"][i].to(device)
                    gt_tr = batch["transcripts"][i]

                    skel = {
                        "segments": [
                            {"begin": seg.get("begin", 0.0), "end": seg.get("end", 0.0)}
                            for seg in gt_tr.get("segments", [])
                        ]
                    }

                    out = model.generate_case(
                        fixations=case,
                        transcript=skel,
                        encode_text_fn=lambda s: vocab.encode(s, add_eos=True),
                        image_1chw=img,
                        max_len=decode_len,
                    )

                    pred_text = " ".join(vocab.decode(t.tolist()) for t in out["gen_tokens_per_segment"]).strip()
                    ref_text = " ".join(seg.get("text", "") for seg in gt_tr.get("segments", []))

                    text_wer_sum += wer(ref_text, pred_text)
                    text_cer_sum += cer(ref_text, pred_text)
                    n_cases += 1
                    exact_match_total += int(pred_text == ref_text)

                    if HAS_NLTK and sentence_bleu is not None:
                        generated_segments = [
                            vocab.decode(t.tolist()) for t in out["gen_tokens_per_segment"]
                        ]
                        reference_segments = [seg.get("text", "") for seg in gt_tr.get("segments", [])]
                        for gen_seg, ref_seg in zip(generated_segments, reference_segments):
                            ref_tokens = ref_seg.split()
                            hyp_tokens = gen_seg.split()
                            if not ref_tokens and not hyp_tokens:
                                bleu = 1.0
                            elif not ref_tokens:
                                bleu = 0.0
                            else:
                                bleu = sentence_bleu(
                                    [ref_tokens],
                                    hyp_tokens,
                                    smoothing_function=bleu_smoothing,
                                )
                            bleu_sum += bleu
                            bleu_count += 1

                    if HAS_ROUGE and rouge_scorer is not None:
                        scores = rouge_scorer.score(ref_text, pred_text)
                        for key in rouge_sums:
                            rouge_sums[key] += scores[key].fmeasure
                        rouge_count += 1

                    fwd = model.forward_case(
                        fixations=case,
                        transcript=gt_tr,
                        encode_text_fn=lambda s: vocab.encode(s, add_eos=True),
                        image_1chw=img,
                    )
                    for logits, seg in zip(fwd["txt_logits_per_segment"], gt_tr.get("segments", [])):
                        if logits.numel() == 0:
                            continue
                        tgt = vocab.encode(seg.get("text", ""), add_eos=True).to(device)
                        tok_nll_sum += F.cross_entropy(logits, tgt, reduction="sum").item()
                        tok_count += tgt.numel()

        avg_wer = text_wer_sum / max(1, n_cases)
        avg_cer = text_cer_sum / max(1, n_cases)
        ppl = math.exp(tok_nll_sum / tok_count) if tok_count > 0 else float("nan")
        exact_match_rate = exact_match_total / max(1, n_cases)
        print(f"Text metrics → WER: {avg_wer:.3f} | CER: {avg_cer:.3f} | PPL: {ppl:.2f} | Exact Match: {exact_match_rate:.3f}")

        if HAS_NLTK and bleu_count > 0:
            avg_bleu = bleu_sum / bleu_count
            print(f"  BLEU (segment avg): {avg_bleu:.3f}")
        else:
            print("  BLEU metric skipped (nltk not installed or no segments).")

        if HAS_ROUGE and rouge_count > 0:
            avg_rouge = {k: v / rouge_count for k, v in rouge_sums.items()}
            print(
                "  ROUGE → "
                + ", ".join(f"{k}: {avg_rouge[k]:.3f}" for k in sorted(avg_rouge.keys()))
            )
        else:
            print("  ROUGE metrics skipped (rouge_score not installed or no samples).")

    if model.decoder is not None:
        # Show a single sample prediction for inspection
        print("\nGenerating sample prediction...")
        sample_batch = next(iter(test_loader))
        fixations = sample_batch["fixations"]
        length = int(fixations["lengths"][0].item())
        case = {
            "xy": fixations["xy"][0, :length].to(device),
            "dwell": fixations["dwell"][0, :length].to(device),
            "time": fixations["time"][0, :length].to(device),
            "seg_hits": fixations["seg_hits"][0, :length].to(device),
            "box_hits": fixations["box_hits"][0, :length].to(device),
        }
        outputs = model.generate_case(
            fixations=case,
            transcript=sample_batch["transcripts"][0],
            encode_text_fn=lambda s: vocab.encode(s, add_eos=True),
            image_1chw=sample_batch["images"][0].to(device),
            max_len=decode_len,
        )
        decoded_segments = [vocab.decode(tokens.tolist()) for tokens in outputs["gen_tokens_per_segment"]]
        print(
            json.dumps(
                {
                    "dicom_id": sample_batch["dicom_ids"][0],
                    "label_probs": torch.sigmoid(outputs["label_logits"]).cpu().tolist(),
                    "segments": decoded_segments,
                },
                indent=2,
            )
        )
    else:
        print("\nTranscript decoder disabled; skipping sample prediction output.")


if __name__ == "__main__":
    main()
