#!/usr/bin/env python3
"""
Aggregate EGD-CXR transcript metadata into a structured directory and CSV.

This utility reads the `audio_segmentation_transcripts/<dicom_id>/transcript.json`
files shipped with the dataset and writes a cleaned transcript tree (JSON + TXT per
case) together with a consolidated `transcripts.csv` suitable for downstream use.

Example:
    python prepare_transcripts.py --config-path config/data_egd-cxr.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from egd_cxr_dataset import ConfigLoader

DEFAULT_CONFIG = ROOT / "config" / "data_egd-cxr.yaml"
DEFAULT_RAW_SUFFIX = "audio_segmentation_transcripts"


@dataclass
class TranscriptRecord:
    dicom_id: str
    transcript: str
    segment_count: int
    has_timestamps: bool
    source_path: Path
    output_json: Path
    output_txt: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare transcript text/JSON assets for the EGD-CXR dataset."
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to configuration YAML (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        help=(
            "Directory containing per-case transcript.json files. "
            "Defaults to <input_path.gaze_raw>/audio_segmentation_transcripts."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where the processed transcripts will be saved. Defaults to input_path.transcripts_dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing transcript files in the output directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N cases (useful for smoke tests).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse transcripts and print the summary without writing files.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Dict[str, Path]:
    config_loader = ConfigLoader(args.config_path)

    raw_dir = args.raw_dir
    if raw_dir is None:
        gaze_root = config_loader.get("input_path", "gaze_raw")
        if gaze_root is None:
            raise ValueError("Configuration missing 'input_path.gaze_raw'.")
        raw_dir = Path(gaze_root).expanduser() / DEFAULT_RAW_SUFFIX

    output_dir = args.output_dir
    if output_dir is None:
        transcripts_dir = config_loader.get("input_path", "transcripts_dir")
        if transcripts_dir is None:
            raise ValueError("Configuration missing 'input_path.transcripts_dir'.")
        output_dir = Path(transcripts_dir).expanduser()

    return {"raw_dir": Path(raw_dir), "output_dir": Path(output_dir)}


def iter_transcript_cases(raw_dir: Path) -> Iterable[Path]:
    for path in sorted(raw_dir.iterdir()):
        if path.is_dir():
            yield path


def load_transcript_json(transcript_path: Path) -> Optional[Dict]:
    if not transcript_path.exists():
        return None
    try:
        return json.loads(transcript_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {transcript_path}: {exc}") from exc


def normalise_segments(payload: Dict) -> List[Dict]:
    segments = payload.get("time_stamped_text") or payload.get("segments") or []
    cleaned: List[Dict] = []
    for entry in segments:
        if not isinstance(entry, dict):
            continue
        cleaned.append(
            {
                "begin": float(entry.get("begin_time", entry.get("start", 0.0))),
                "end": float(entry.get("end_time", entry.get("end", 0.0))),
                "text": str(entry.get("phrase", entry.get("text", ""))).strip(),
            }
        )
    return cleaned


def write_case_outputs(
    dicom_id: str,
    transcript_text: str,
    segments: List[Dict],
    source_path: Path,
    output_dir: Path,
    overwrite: bool,
) -> TranscriptRecord:
    case_dir = output_dir / dicom_id
    case_dir.mkdir(parents=True, exist_ok=True)

    json_path = case_dir / "transcript.json"
    txt_path = case_dir / "transcript.txt"

    payload = {
        "dicom_id": dicom_id,
        "transcript": transcript_text,
        "segments": segments,
        "source_json": str(source_path),
    }

    if overwrite or not json_path.exists():
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if overwrite or not txt_path.exists():
        txt_path.write_text((transcript_text or "") + ("\n" if transcript_text else ""), encoding="utf-8")

    return TranscriptRecord(
        dicom_id=dicom_id,
        transcript=transcript_text,
        segment_count=len(segments),
        has_timestamps=bool(segments),
        source_path=source_path,
        output_json=json_path,
        output_txt=txt_path,
    )


def process_transcripts(
    raw_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool,
    limit: Optional[int],
    dry_run: bool,
) -> List[TranscriptRecord]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw transcript directory not found: {raw_dir}")

    cases = list(iter_transcript_cases(raw_dir))
    if limit is not None:
        cases = cases[:limit]

    records: List[TranscriptRecord] = []
    missing = 0

    for case_dir in cases:
        dicom_id = case_dir.name
        transcript_path = case_dir / "transcript.json"
        payload = load_transcript_json(transcript_path)
        if payload is None:
            missing += 1
            continue

        transcript_text = str(payload.get("full_text") or payload.get("transcript") or "").strip()
        segments = normalise_segments(payload)

        if dry_run:
            records.append(
                TranscriptRecord(
                    dicom_id=dicom_id,
                    transcript=transcript_text,
                    segment_count=len(segments),
                    has_timestamps=bool(segments),
                    source_path=transcript_path,
                    output_json=output_dir / dicom_id / "transcript.json",
                    output_txt=output_dir / dicom_id / "transcript.txt",
                )
            )
            continue

        record = write_case_outputs(
            dicom_id,
            transcript_text,
            segments,
            transcript_path,
            output_dir,
            overwrite=overwrite,
        )
        records.append(record)

    if missing:
        print(f"⚠ Skipped {missing} cases without transcript.json")

    return records


def write_summary_csv(records: List[TranscriptRecord], output_dir: Path, overwrite: bool, dry_run: bool) -> None:
    if dry_run:
        return
    if not records:
        print("No transcripts processed; skipping CSV.")
        return

    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "dicom_id": rec.dicom_id,
                "transcript": rec.transcript,
                "segment_count": rec.segment_count,
                "has_timestamps": rec.has_timestamps,
                "source_json": str(rec.source_path),
                "output_json": str(rec.output_json),
                "output_txt": str(rec.output_txt),
            }
            for rec in records
        ]
    ).sort_values("dicom_id")

    csv_path = output_dir / "transcripts.csv"
    if overwrite or not csv_path.exists():
        df.to_csv(csv_path, index=False)
    else:
        print(f"Skipping CSV write (exists and overwrite disabled): {csv_path}")


def print_summary(records: List[TranscriptRecord], output_dir: Path) -> None:
    if not records:
        print("No transcripts processed.")
        return

    total_tokens = sum(len(rec.transcript.split()) for rec in records)
    print(
        f"Processed {len(records)} transcripts -> output dir: {output_dir}\n"
        f"  With timestamps: {sum(rec.has_timestamps for rec in records)}\n"
        f"  Total tokens: {total_tokens}"
    )


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)
    raw_dir = paths["raw_dir"]
    output_dir = paths["output_dir"]

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    records = process_transcripts(
        raw_dir,
        output_dir,
        overwrite=args.overwrite,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    write_summary_csv(records, output_dir, args.overwrite, args.dry_run)
    print_summary(records, output_dir)


if __name__ == "__main__":
    main()
