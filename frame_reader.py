import cv2
from typing import Callable

from util import frame_diff, extract_interesting_words


class FrameGroup:
    """
    Holds all information for one visual group (consecutive frames showing the same slide).

    init args: start_time, end_time, first_continuous_text, frame, words_on_slide,
               first_important_words_said, diff_threshold
    """

    def __init__(
        self,
        start_time: float,
        end_time: float,
        first_continuous_text: str,
        frame,           # (H, W) uint8 grayscale, representative of this group
        words_on_slide: list,
        first_important_words_said: list,
        diff_threshold: float = 1.0,
    ):
        self.start_time = start_time
        self.end_time = end_time
        self.continuous_text = first_continuous_text
        self.frame = frame
        self.words_on_slide = list(words_on_slide)
        self.important_words_said = list(first_important_words_said)
        self._diff_threshold = diff_threshold

    def add(self, text_said: str, important_words_said: list, new_end_time: float):
        """Merge a new segment into this group."""
        existing = set(self.important_words_said)
        self.important_words_said.extend(w for w in important_words_said if w not in existing)
        self.continuous_text += ' ' + text_said
        if new_end_time > self.end_time:
            self.end_time = new_end_time

    def belongs_to_this(self, frame) -> bool:
        """
        True when mean absolute pixel difference between frame and the representative
        is below the configured threshold.  None frames never belong.
        """
        if self.frame is None or frame is None:
            return False
        diff = frame_diff(self.frame, frame)
        return diff < self._diff_threshold


class FrameReader:
    """
    Consumes pre-extracted frames one at a time and accumulates them into FrameGroup objects.

    crop_coords : ((x1, y1), (x2, y2)) — bounding box; use None for x2/y2 to crop to image edge.
    raw frames  : ndarray (BGR or grayscale) — reading the frame is the caller's responsibility.
    ocr_hook    : optional callable(cropped_frame) -> list[str]  (words found on the slide).
                  Receives the cropped, pre-grayscale frame.
    """

    def __init__(
        self,
        crop_coords,
        target_size: tuple = (128, 128),
        ocr_hook: Callable = None,
        diff_threshold: float = 1.0,
    ):
        (ax, ay), (bx, by) = crop_coords
        self._x1 = min(v for v in [ax, bx] if v is not None) if any(v is not None for v in [ax, bx]) else 0
        self._y1 = min(v for v in [ay, by] if v is not None) if any(v is not None for v in [ay, by]) else 0
        self._x2 = max(v for v in [ax, bx] if v is not None) if all(v is not None for v in [ax, bx]) else None
        self._y2 = max(v for v in [ay, by] if v is not None) if all(v is not None for v in [ay, by]) else None

        self._target_size = target_size
        self._ocr_hook = ocr_hook
        self._diff_threshold = diff_threshold

        self.group_list: list[FrameGroup] = []
        self._current_group: FrameGroup = None

    def _extract_interesting_words(self, text: str) -> list:
        return extract_interesting_words(text)

    def add_new_element(self, timestamp, raw_frame, text_said: str):
        """
        Process one segment.

        timestamp   : object with .start and .end attributes (seconds).
        raw_frame   : (H, W, C) BGR ndarray, or None if extraction failed.
        text_said   : raw transcript text for this segment.
        """
        if raw_frame is None:
            resized = None
            words_on_slide = []
        else:
            h, w = raw_frame.shape[:2]
            x2 = self._x2 if self._x2 is not None else w
            y2 = self._y2 if self._y2 is not None else h
            cropped = raw_frame[self._y1:y2, self._x1:x2]

            # OCR on full-resolution cropped frame before any downscaling
            words_on_slide = self._ocr_hook(cropped) if self._ocr_hook else []

            # Grayscale
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) if cropped.ndim == 3 else cropped

            # Resize to fixed target size
            resized = cv2.resize(gray, self._target_size, interpolation=cv2.INTER_LANCZOS4)

        important_words_said = self._extract_interesting_words(text_said)

        if self._current_group is not None and self._current_group.belongs_to_this(resized):
            self._current_group.add(text_said, important_words_said, timestamp.end)
        else:
            if self._current_group is not None:
                self.group_list.append(self._current_group)
            self._current_group = FrameGroup(
                start_time=timestamp.start,
                end_time=timestamp.end,
                first_continuous_text=text_said,
                frame=resized,
                words_on_slide=words_on_slide,
                first_important_words_said=important_words_said,
                diff_threshold=self._diff_threshold,
            )

    def close_stream(self):
        """Flush the last open group into group_list."""
        if self._current_group is not None:
            self.group_list.append(self._current_group)
            self._current_group = None
