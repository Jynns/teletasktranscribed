"""
Lecture video transcription using ffmpeg + faster-whisper.

Usage:
    python transcribe.py <video.mp4> [--model tiny|base|small|medium|large-v3] [--device auto|cpu|cuda]
    python transcribe.py data/lecture.mp4
    python transcribe.py data/lecture.mp4 --model large-v3 --device cuda
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if sys.platform == "win32":
    # The pip-installed nvidia-cublas-cu12 / nvidia-cudnn-cu12 wheels ship their DLLs
    # inside site-packages instead of a system CUDA install; make them loadable.
    # ctranslate2 resolves them via the legacy LoadLibrary search (PATH), so
    # add_dll_directory alone isn't sufficient -- PATH must be extended too.
    import os
    for _pkg in ("cublas", "cudnn"):
        _dll_dir = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / _pkg / "bin"
        if _dll_dir.is_dir():
            os.add_dll_directory(str(_dll_dir))
            os.environ["PATH"] = str(_dll_dir) + os.pathsep + os.environ["PATH"]

import cv2
import fitz
import numpy as np
from faster_whisper import WhisperModel

from assignment_model import AssignmentModel, SlideHandler, export_script
from frame_reader import FrameReader
from util import find_aoi_corner, format_timestamp


def extract_audio(video_path: Path, audio_path: Path) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)


def _extract_frame_bgr(video_path: Path, timestamp: float) -> np.ndarray | None:
    """Return a full-resolution BGR frame at timestamp, or None on failure."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(timestamp),
        "-i", str(video_path),
        "-frames:v", "1",
        "-f", "image2pipe",
        "-vcodec", "png",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not result.stdout:
        return None
    arr = np.frombuffer(result.stdout, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def calibrate_crop_corner(seg_list, video_path: Path, n: int = 10) -> tuple[int, int]:
    """
    Sample n evenly-spaced segments, detect the AOI corner in each frame,
    and return the median (x, y) as the stable crop corner.
    """
    indices = np.linspace(0, len(seg_list) - 1, n, dtype=int)
    x_vals: list[int] = []
    y_vals: list[int] = []
    for i in indices:
        frame_bgr = _extract_frame_bgr(video_path, seg_list[i].start)
        if frame_bgr is None:
            continue
        x, y = find_aoi_corner(frame_bgr)
        if x is not None:
            x_vals.append(x)
        if y is not None:
            y_vals.append(y)
    x_crop = int(np.median(x_vals)) if x_vals else 0
    y_crop = int(np.median(y_vals)) if y_vals else 0
    print(f"  AOI corner calibrated from {len(x_vals)}/{n} frames: x={x_crop}, y={y_crop}")
    return x_crop, y_crop


def _load_slide_images(pdf_path: Path) -> tuple[list[np.ndarray], list[str]]:
    """Render each PDF page to BGR and extract embedded text via fitz."""
    images, texts = [], []
    pdf = fitz.open(str(pdf_path))
    for page in pdf:
        pix = page.get_pixmap(dpi=72)
        rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        images.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        texts.append(page.get_text("text"))
    pdf.close()
    return images, texts


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _load_whisper_model(model_name: str, device: str) -> WhisperModel:
    """Load the model on the requested device, falling back to CPU if CUDA is unavailable.

    compute_type="int8" is used on GPU rather than float16 because Pascal-generation
    cards (e.g. GTX 10-series) lack efficient fp16 throughput.
    """
    if device == "auto":
        try:
            return WhisperModel(model_name, device="cuda", compute_type="int8")
        except Exception as exc:
            print(f"  GPU load failed ({exc}); falling back to CPU.")
            return WhisperModel(model_name, device="cpu", compute_type="int8")
    if device == "cuda":
        return WhisperModel(model_name, device="cuda", compute_type="int8")
    return WhisperModel(model_name, device="cpu", compute_type="int8")


def transcribe(video_path: Path, model_name: str, device: str) -> None:
    audio_path = video_path.with_suffix(".wav")
    output_txt = video_path.with_suffix(".txt")
    output_srt = video_path.with_suffix(".srt")

    print(f"Extracting audio from {video_path.name} ...")
    extract_audio(video_path, audio_path)
    print(f"  -> {audio_path.name}")

    print(f"Loading Whisper model '{model_name}' (first run downloads weights) ...")
    model = _load_whisper_model(model_name, device)

    print("Transcribing ...")
    segments, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        language=None,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    print(f"  Detected language: {info.language} (probability {info.language_probability:.2f})")

    seg_list = []
    for seg in segments:
        seg_list.append(seg)
        print(f"  [{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}]  {seg.text.strip()}")

    with open(output_txt, "w", encoding="utf-8") as f:
        for seg in seg_list:
            f.write(f"[{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}]\n")
            f.write(seg.text.strip() + "\n\n")

    with open(output_srt, "w", encoding="utf-8") as f:
        for i, seg in enumerate(seg_list, start=1):
            start_srt = format_timestamp(seg.start).replace(".", ",")
            end_srt   = format_timestamp(seg.end).replace(".", ",")
            f.write(f"{i}\n{start_srt} --> {end_srt}\n{seg.text.strip()}\n\n")

    print(f"\nTranscript saved to:  {output_txt}")
    print(f"SRT subtitles saved to: {output_srt}")

    pdf_path = video_path.with_suffix(".pdf")
    if not pdf_path.exists():
        print(f"\nNo PDF found at {pdf_path} — skipping slide mapping.")
        audio_path.unlink()
        return

    # ── Slides ───────────────────────────────────────────────────────────────
    print(f"\nLoading slides from {pdf_path.name} ...")
    slide_images, slide_texts = _load_slide_images(pdf_path)
    slide_handler = SlideHandler(slide_images, slide_texts=slide_texts)
    slides = slide_handler.slides
    print(f"  {len(slides)} slides loaded")

    # ── Frame grouping ───────────────────────────────────────────────────────
    print("Calibrating slide area of interest ...")
    crop_x, crop_y = calibrate_crop_corner(seg_list, video_path)

    print(f"Extracting and grouping {len(seg_list)} frames ...")
    reader = FrameReader(
        crop_coords=((crop_x, crop_y), (None, None)),
        target_size=(128, 128),
    )
    total = len(seg_list)
    for i, seg in enumerate(seg_list, start=1):
        bgr = _extract_frame_bgr(video_path, seg.start)
        reader.add_new_element(seg, bgr, seg.text)
        print(f"\r  {i}/{total} ({100*i//total}%)", end="", flush=True)
    print()
    reader.close_stream()
    groups = reader.group_list
    print(f"  {len(seg_list)} segments -> {len(groups)} groups")

    # ── HMM assignment ───────────────────────────────────────────────────────
    print("Assigning slides via HMM forward-backward ...")
    hmm = AssignmentModel(slides)
    assignment = hmm.assign_slides(groups)

    # ── Export ───────────────────────────────────────────────────────────────
    result_dir = Path.cwd() / "result"
    result_dir.mkdir(exist_ok=True)
    output_slides = result_dir / video_path.with_suffix(".txt").name
    export_script(assignment, slides, groups, output_slides)
    print(f"Slide transcript saved to: {output_slides}")

    audio_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe a lecture video to text.")
    parser.add_argument("video", type=Path, help="Path to the input .mp4 file")
    parser.add_argument(
        "--model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper model size (default: small). Larger = more accurate but slower.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device (default: auto — tries GPU, falls back to CPU).",
    )
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    transcribe(args.video, args.model, args.device)


if __name__ == "__main__":
    main()
