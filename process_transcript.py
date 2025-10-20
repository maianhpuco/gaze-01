#!/usr/bin/env python3
"""
Transcribe radiologist dictation audio for a given EGD-CXR case using OpenAI Whisper.

The script searches for the audio file associated with the provided case ID inside the
EGD-CXR dataset hierarchy and produces both plain-text and JSON transcripts. By default,
it targets the same exemplar case ID used throughout this repository so running:

    python process_transcript.py

is sufficient once the dataset and Whisper model are available locally.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path("/project/hnguyen2/mvu9/datasets/gaze_data/physionet.org/files/egd-cxr/1.0.0")
DEFAULT_AUDIO_ROOT = DEFAULT_DATA_ROOT / "audio_segmentation_transcripts"
DEFAULT_OUTPUT_ROOT = ROOT / "plots" / "transcripts"
DEFAULT_CASE_ID = "24c7496c-d7635dfe-b8e0b87f-d818affc-78ff7cf4"
DEFAULT_MODEL_NAME = "base"
SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg")


@dataclass
class TranscriptResult:
    text: str
    segments: List[dict]
    language: Optional[str]
    duration_seconds: Optional[float]
    audio_path: Path
    model_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe radiologist dictation audio using Whisper."
    )
    parser.add_argument(
        "--case-id",
        default=DEFAULT_CASE_ID,
        help=f"Case identifier to process (default: {DEFAULT_CASE_ID}).",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=DEFAULT_AUDIO_ROOT,
        help=f"Directory containing the audio_segmentation_transcripts/ tree (default: {DEFAULT_AUDIO_ROOT}).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Directory where transcripts will be stored (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Whisper model name to load (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "--language",
        help="Optional language hint for Whisper (e.g. 'en'). If omitted, Whisper auto-detects.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for Whisper inference: 'auto', 'cpu', or CUDA device spec (e.g. 'cuda', 'cuda:1').",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging from Whisper during transcription.",
    )
    return parser.parse_args()


def resolve_audio_file(case_id: str, audio_root: Path) -> Path:
    case_dir = audio_root / case_id
    if not case_dir.is_dir():
        raise FileNotFoundError(f"Could not find audio directory for case {case_id} under {audio_root}")

    direct_match_files = [
        case_dir / f"{case_id}{ext}" for ext in SUPPORTED_AUDIO_EXTENSIONS if (case_dir / f"{case_id}{ext}").exists()
    ]
    if direct_match_files:
        return direct_match_files[0]

    candidates = sorted(
        [p for p in case_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS]
    )
    if not candidates:
        raise FileNotFoundError(
            f"No audio files with extensions {SUPPORTED_AUDIO_EXTENSIONS} found in {case_dir}"
        )
    return candidates[0]


def load_whisper_model(model_name: str, device: str):
    try:
        import whisper  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "The 'whisper' package is required. Install it via 'pip install -U openai-whisper'."
        ) from exc

    if device == "auto":
        return whisper.load_model(model_name)
    return whisper.load_model(model_name, device=device)


def transcribe_audio(
    model,
    audio_path: Path,
    language: Optional[str],
    verbose: bool,
) -> TranscriptResult:
    kwargs = {"fp16": False} if "cpu" in str(model.device).lower() else {}
    if language:
        kwargs["language"] = language
    if verbose:
        kwargs["verbose"] = True

    result = model.transcribe(str(audio_path), **kwargs)
    segments = [
        {
            "id": segment["id"],
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "text": segment["text"],
        }
        for segment in result.get("segments", [])
    ]
    return TranscriptResult(
        text=result.get("text", "").strip(),
        segments=segments,
        language=result.get("language"),
        duration_seconds=float(result.get("duration")) if result.get("duration") is not None else None,
        audio_path=audio_path,
        model_name=model.name if hasattr(model, "name") else "whisper",
    )


def save_transcript(output_root: Path, case_id: str, transcript: TranscriptResult) -> None:
    case_output_dir = output_root / case_id
    case_output_dir.mkdir(parents=True, exist_ok=True)

    text_path = case_output_dir / "transcript.txt"
    json_path = case_output_dir / "transcript.json"

    text_path.write_text(transcript.text or "", encoding="utf-8")

    payload = asdict(transcript)
    payload["segments"] = transcript.segments  # asdict already handles it but we make intent clear
    payload["audio_path"] = str(transcript.audio_path)
    payload["output_text_path"] = str(text_path)
    payload["output_json_path"] = str(json_path)

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote transcript text to {text_path}")
    print(f"Wrote transcript metadata to {json_path}")


def main() -> None:
    args = parse_args()

    audio_path = resolve_audio_file(args.case_id, args.audio_root)
    print(f"Found audio file: {audio_path}")

    try:
        model = load_whisper_model(args.model, args.device)
    except RuntimeError as err:
        print(err, file=sys.stderr)
        sys.exit(1)

    print(f"Loaded Whisper model '{args.model}' on device '{args.device}'")

    transcript = transcribe_audio(model, audio_path, args.language, args.verbose)
    save_transcript(args.output_root, args.case_id, transcript)


if __name__ == "__main__":
    main()
