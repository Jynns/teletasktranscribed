"""
Lecture video transcription using ffmpeg + faster-whisper.

Usage:
    python transcribe.py <video.mp4> [--model tiny|base|small|medium|large-v3]
    python transcribe.py data/lecture.mp4
    python transcribe.py data/lecture.mp4 --model large-v3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import fitz
import numpy as np
from faster_whisper import WhisperModel


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


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


_TARGET_SIZE = (128, 128)


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Normalization (shared by frames and slides)
# ---------------------------------------------------------------------------

def _normalize_frame(gray: np.ndarray) -> np.ndarray:
    """Resize a grayscale uint8 array to TARGET_SIZE and apply mean/std normalization."""
    resized = cv2.resize(gray, _TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
    arr = resized.astype(np.float32)
    mean, std = arr.mean(), arr.std()
    return (arr - mean) / (std if std > 0 else 1.0)


# ---------------------------------------------------------------------------
# Area-of-interest calibration (double-pass Sobel + Hough)
# ---------------------------------------------------------------------------

def _find_aoi_corner(
    frame_bgr: np.ndarray,
    ksize: int = 3,
    threshold: int = 80,
    angle_tol: int = 6,
) -> tuple[int | None, int | None]:
    """
    Detect the upper-left corner of the slide area in a full-res BGR frame.
    Returns (x_from_longest_vertical_line, y_from_longest_horizontal_line).
    The transformation applied here is purely for detection — it is discarded afterwards.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    sx  = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=ksize)
    sy  = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=ksize)
    sx2 = cv2.Sobel(np.abs(sx), cv2.CV_64F, 1, 0, ksize=ksize)
    sy2 = cv2.Sobel(np.abs(sy), cv2.CV_64F, 0, 1, ksize=ksize)

    bx = cv2.threshold(cv2.convertScaleAbs(sx + 3 * sx2), threshold, 255, cv2.THRESH_BINARY)[1]
    by = cv2.threshold(cv2.convertScaleAbs(sy + 3 * sy2), threshold, 255, cv2.THRESH_BINARY)[1]

    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    closed = cv2.bitwise_or(
        cv2.morphologyEx(bx, cv2.MORPH_CLOSE, k_v),
        cv2.morphologyEx(by, cv2.MORPH_CLOSE, k_h),
    )

    lines = cv2.HoughLinesP(
        closed, rho=1, theta=np.pi / 180, threshold=40,
        minLineLength=40, maxLineGap=50,
    )

    h_lines: list[tuple[float, np.ndarray]] = []
    v_lines: list[tuple[float, np.ndarray]] = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.hypot(x2 - x1, y2 - y1)
            if abs(y2 - y1) <= angle_tol:
                h_lines.append((length, line[0]))
            elif abs(x2 - x1) <= angle_tol:
                v_lines.append((length, line[0]))

    h_lines.sort(key=lambda t: -t[0])
    v_lines.sort(key=lambda t: -t[0])

    x_crop = int((v_lines[0][1][0] + v_lines[0][1][2]) / 2) if v_lines else None
    y_crop = int((h_lines[0][1][1] + h_lines[0][1][3]) / 2) if h_lines else None
    return x_crop, y_crop


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
        x, y = _find_aoi_corner(frame_bgr)
        if x is not None:
            x_vals.append(x)
        if y is not None:
            y_vals.append(y)
    x_crop = int(np.median(x_vals)) if x_vals else 0
    y_crop = int(np.median(y_vals)) if y_vals else 0
    print(f"  AOI corner calibrated from {len(x_vals)}/{n} frames: x={x_crop}, y={y_crop}")
    return x_crop, y_crop


# ---------------------------------------------------------------------------
# Slide loading and frame extraction (final, for matching)
# ---------------------------------------------------------------------------

def load_pdf_slides(pdf_path: Path) -> list[np.ndarray]:
    """Load all PDF pages as normalized grayscale arrays (no crop)."""
    slides = []
    pdf = fitz.open(str(pdf_path))
    for page in pdf:
        pix = page.get_pixmap(dpi=72)
        rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        slides.append(_normalize_frame(gray))
    pdf.close()
    return slides


def extract_frame(
    video_path: Path,
    timestamp: float,
    crop: tuple[int, int] | None = None,
) -> np.ndarray | None:
    """Extract a frame, optionally crop to the slide AOI, then normalize."""
    frame_bgr = _extract_frame_bgr(video_path, timestamp)
    if frame_bgr is None:
        return None
    if crop is not None:
        x_crop, y_crop = crop
        h, w = frame_bgr.shape[:2]
        frame_bgr = frame_bgr[y_crop:h, x_crop:w]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return _normalize_frame(gray)


def best_slide(frame: np.ndarray, slides: list[np.ndarray]) -> int:
    diffs = [np.sum(np.abs(frame - s)) for s in slides]
    return int(np.argmin(diffs)) + 1  # 1-based


def map_segments_to_slides(
    seg_list,
    video_path: Path,
    slides: list[np.ndarray],
    crop: tuple[int, int] | None = None,
) -> list[int]:
    assignments = []
    for seg in seg_list:
        frame = extract_frame(video_path, seg.start, crop)
        if frame is not None:
            assignments.append(best_slide(frame, slides))
        else:
            assignments.append(assignments[-1] if assignments else 1)
    return assignments


# ---------------------------------------------------------------------------
# Slide transcript: consecutive-window merging
# ---------------------------------------------------------------------------

def build_slide_transcript(
    seg_list,
    assignments: list[int],
) -> list[tuple[int, float, float, list[str]]]:
    """
    Group segments into consecutive windows: a new window opens whenever the
    assigned slide changes, even if that slide was seen before.

    Returns a list of (slide_num, window_start, window_end, [texts])
    ordered by window_start.
    """
    pairs = sorted(zip(seg_list, assignments), key=lambda p: p[0].start)
    windows: list[tuple[int, float, float, list[str]]] = []
    for seg, slide_num in pairs:
        if windows and windows[-1][0] == slide_num:
            num, start, end, texts = windows[-1]
            windows[-1] = (num, start, max(end, seg.end), texts + [seg.text.strip()])
        else:
            windows.append((slide_num, seg.start, seg.end, [seg.text.strip()]))
    return windows


def save_slide_transcript(
    windows: list[tuple[int, float, float, list[str]]],
    output_path: Path,
) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for slide_num, start, end, texts in windows:
            f.write(f"slide #{slide_num} [{format_timestamp(start)} --> {format_timestamp(end)}]\n")
            f.write(" ".join(texts) + "\n\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def transcribe(video_path: Path, model_name: str) -> None:
    audio_path = video_path.with_suffix(".wav")
    output_txt = video_path.with_suffix(".txt")
    output_srt = video_path.with_suffix(".srt")

    print(f"Extracting audio from {video_path.name} ...")
    extract_audio(video_path, audio_path)
    print(f"  -> {audio_path.name}")

    print(f"Loading Whisper model '{model_name}' (first run downloads weights) ...")
    model = WhisperModel(model_name, device="cpu", compute_type="int8")

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
            end_srt = format_timestamp(seg.end).replace(".", ",")
            f.write(f"{i}\n{start_srt} --> {end_srt}\n{seg.text.strip()}\n\n")

    print(f"\nTranscript saved to:  {output_txt}")
    print(f"SRT subtitles saved to: {output_srt}")

    pdf_path = video_path.with_suffix(".pdf")
    if pdf_path.exists():
        print(f"\nLoading slides from {pdf_path.name} ...")
        slides = load_pdf_slides(pdf_path)
        print(f"  -> {len(slides)} slides loaded")

        print("Calibrating slide area of interest from 10 frames ...")
        crop = calibrate_crop_corner(seg_list, video_path)

        print("Mapping segments to slides ...")
        assignments = map_segments_to_slides(seg_list, video_path, slides, crop)
        for seg, slide_num in zip(seg_list, assignments):
            print(f"  [{format_timestamp(seg.start)}]  -> slide {slide_num}")

        windows = build_slide_transcript(seg_list, assignments)
        output_slides = video_path.with_stem(video_path.stem + "_slides").with_suffix(".txt")
        save_slide_transcript(windows, output_slides)
        print(f"Slide transcript saved to: {output_slides}")
    else:
        print(f"\nNo PDF found at {pdf_path} — skipping slide mapping.")

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
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    transcribe(args.video, args.model)


if __name__ == "__main__":
    main()
