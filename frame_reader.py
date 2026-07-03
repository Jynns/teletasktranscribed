import cv2

try:
    import pytesseract
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False

from util import frame_diff, extract_interesting_words, ocr_crop


class FrameGroup:
    """
    Holds all information for one visual group (consecutive frames showing the same slide).

    frame_representetive : full-resolution grayscale crop — used for grouping comparison
                           and for visualisation.
    frame_transformed    : resized (target_size) uint8 grayscale — used for emission.
    """

    def __init__(
        self,
        start_time: float,
        end_time: float,
        first_continuous_text: str,
        frame_transformed,
        words_on_slide: list,
        first_important_words_said: list,
        frame_representetive,
        diff_threshold: float = 1.0,
    ):
        self.start_time = start_time
        self.end_time = end_time
        self.continuous_text = first_continuous_text
        self.frame_transformed = frame_transformed
        self.words_on_slide = list(words_on_slide)
        self.important_words_said = list(first_important_words_said)
        self._diff_threshold = diff_threshold
        self.frame_representetive = frame_representetive

    def add(self, text_said: str, important_words_said: list, new_end_time: float):
        existing = set(self.important_words_said)
        self.important_words_said.extend(w for w in important_words_said if w not in existing)
        self.continuous_text += ' ' + text_said
        if new_end_time > self.end_time:
            self.end_time = new_end_time

    def belongs_to_this(self, frame) -> bool:
        if self.frame_representetive is None or frame is None:
            return False
        return frame_diff(self.frame_representetive, frame) < self._diff_threshold


class FrameReader:
    """
    Consumes pre-extracted frames one at a time and accumulates them into FrameGroup objects.

    crop_coords : ((x1, y1), (x2, y2)) — bounding box; use None for x2/y2 to crop to edge.
    raw frames  : BGR ndarray — reading the frame is the caller's responsibility.
    """

    def __init__(
        self,
        crop_coords,
        target_size: tuple = (128, 128),
        diff_threshold: float = 1.0,
    ):
        (ax, ay), (bx, by) = crop_coords
        self._x1 = min(v for v in [ax, bx] if v is not None) if any(v is not None for v in [ax, bx]) else 0
        self._y1 = min(v for v in [ay, by] if v is not None) if any(v is not None for v in [ay, by]) else 0
        self._x2 = max(v for v in [ax, bx] if v is not None) if all(v is not None for v in [ax, bx]) else None
        self._y2 = max(v for v in [ay, by] if v is not None) if all(v is not None for v in [ay, by]) else None

        self._target_size = target_size
        self._diff_threshold = diff_threshold

        self.group_list: list[FrameGroup] = []
        self._current_group: FrameGroup = None

    def _extract_interesting_words(self, text: str) -> list:
        return extract_interesting_words(text)

    def add_new_element(self, timestamp, raw_frame, text_said: str):
        """
        timestamp  : object with .start and .end (seconds).
        raw_frame  : BGR ndarray or None.
        text_said  : raw transcript text for this segment.
        """
        important_words_said = self._extract_interesting_words(text_said)

        if raw_frame is None:
            # No visual data — always starts a new group, no frame representative
            if self._current_group is not None:
                self.group_list.append(self._current_group)
            self._current_group = FrameGroup(
                start_time=timestamp.start,
                end_time=timestamp.end,
                first_continuous_text=text_said,
                frame_transformed=None,
                words_on_slide=[],
                first_important_words_said=important_words_said,
                frame_representetive=None,
                diff_threshold=self._diff_threshold,
            )
            return

        h, w = raw_frame.shape[:2]
        x2 = self._x2 if self._x2 is not None else w
        y2 = self._y2 if self._y2 is not None else h
        cropped = raw_frame[self._y1:y2, self._x1:x2]
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) if cropped.ndim == 3 else cropped

        if self._current_group is not None and self._current_group.belongs_to_this(gray):
            self._current_group.add(text_said, important_words_said, timestamp.end)
        else:
            words_on_slide = ocr_crop(gray)
            resized = cv2.resize(gray, self._target_size, interpolation=cv2.INTER_LANCZOS4)
            if self._current_group is not None:
                self.group_list.append(self._current_group)
            self._current_group = FrameGroup(
                start_time=timestamp.start,
                end_time=timestamp.end,
                first_continuous_text=text_said,
                frame_transformed=resized,
                words_on_slide=words_on_slide,
                first_important_words_said=important_words_said,
                frame_representetive=gray,
                diff_threshold=self._diff_threshold,
            )

    def close_stream(self):
        if self._current_group is not None:
            self.group_list.append(self._current_group)
            self._current_group = None
