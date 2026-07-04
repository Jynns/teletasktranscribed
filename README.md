# videoToTranscript

Transcribes lecture recordings and maps the spoken content to PDF slides using a multimodal Hidden Markov Model.

## What it does

1. Extracts audio from a video file and transcribes it with [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
2. Detects the slide area in the video automatically (Sobel + Hough calibration).
3. Groups consecutive frames that show the same slide into segments.
4. Assigns each segment to a PDF slide via HMM forward-backward, combining three emission signals:
   - **Visual**: inverse L1 pixel distance between the frame and each slide image.
   - **Slide OCR words**: Jaccard IoU between OCR text on the frame and slide text.
   - **Spoken words IoU**: Jaccard IoU between spoken important words and slide vocabulary.
5. Exports a structured transcript with slide numbers and timestamps.

## Output files

| File | Description |
|------|-------------|
| `<name>.txt` | Raw Whisper transcript with timestamps |
| `<name>.srt` | SRT subtitle file |
| `<name>_slides.txt` | Slide-annotated transcript (slide # + spoken text per slide) |

## Setup

```bash
pip install -r requirements.txt
```

Tesseract must be installed separately for OCR fallback on slides without embedded text:

```bash
# macOS
brew install tesseract

# Ubuntu / Debian
sudo apt install tesseract-ocr
```

## Usage

```bash
python transcribe.py <video.mp4> [--model MODEL]
```

The PDF must have the same base name as the video and sit in the same directory:

```
data/
  lecture.mp4
  lecture.pdf      ← required for slide mapping
```

### Model sizes

| Model | Speed | Accuracy |
|-------|-------|----------|
| `tiny` | fastest | lowest |
| `base` | fast | — |
| `small` | *(default)* | good |
| `medium` | slow | better |
| `large-v2` / `large-v3` | slowest | best |

### Examples

```bash
# Default (small model)
python transcribe.py data/lecture.mp4

# Higher accuracy
python transcribe.py data/lecture.mp4 --model large-v3
```

## Slide transcript format

```
slide #3  [00:04:12.500 --> 00:07:45.200]
  So today we'll look at query planning. The optimizer chooses between ...

slide #4  [00:07:45.200 --> 00:11:02.800]
  The cost model estimates I/O and CPU cost for each candidate plan ...
```

## Project structure

```
transcribe.py        — CLI entry point and main pipeline
frame_reader.py      — FrameReader (groups video frames) + FrameGroup
assignment_model.py  — Slide, SlideHandler, EmissionHandler, AssignmentModel, export_script
util.py              — Pure transformation functions (tokenize, OCR, AOI detection, ...)
notebook/
  multimodal_hmm.ipynb  — Development notebook with emission visualizations
```
