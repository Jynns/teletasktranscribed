import cv2
import numpy as np
from scipy.special import logsumexp

from pathlib import Path

from util import extract_interesting_words, ocr_crop, normalize_gray, format_timestamp


class Slide:
    """
    Represents one PDF slide — visual processing only.

    image       : BGR ndarray (already rendered, not a path).
    bag_of_words: set of important words — provided by SlideHandler.
    target_size : (W, H) for the visual embedding.
    """

    def __init__(self, image, bag_of_words: set, target_size: tuple = (128, 128)):
        self.bag_of_words = bag_of_words

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_LANCZOS4)

        self._slide_img_uint8 = resized
        self.slide_img = normalize_gray(resized)

    @property
    def slide_img_uint8(self) -> np.ndarray:
        return self._slide_img_uint8


class SlideHandler:
    """
    Receives pre-rendered BGR slide images, extracts text, and creates Slide instances.

    slide_images : list of BGR ndarrays — already rendered, not paths.
    slide_texts  : optional list of pre-extracted strings (e.g. from fitz).
                   When provided, used instead of OCR for the corresponding slide.
                   Falls back to OCR when the entry is None or too short.
    """

    MIN_TEXT_CHARS = 30

    def __init__(self, slide_images: list, slide_texts: list = None, target_size: tuple = (128, 128)):
        texts = slide_texts if slide_texts is not None else [None] * len(slide_images)
        self.slides: list[Slide] = []
        for image, text in zip(slide_images, texts):
            if text and len(text.strip()) >= self.MIN_TEXT_CHARS:
                words = set(extract_interesting_words(text))
            else:
                words = set(ocr_crop(image))
            self.slides.append(Slide(image, bag_of_words=words, target_size=target_size))


class EmissionHandler:
    """
    Computes joint emission probability P(group | slide) for all slides.

    Three modalities, each returning a normalised probability vector over slides:
      - spoken words  : Bernoulli over slide vocabulary (2/3 hit, 1/3 miss)
      - picture       : inverse L1 pixel distance between group frame and slide image
      - slide words   : IoU between OCR words on the frame and slide vocabulary

    get_log_emission(group) multiplies all three and returns the log-normalised result.
    """

    def __init__(self, slides: list):
        self.slides = slides
        self._LOG_IN  = np.log(2 / 3)
        self._LOG_OUT = np.log(1 / 3)

    def _emit_spoken_words(self, group) -> np.ndarray:
        """
        For each slide s, sum log(2/3) for every word in s.bag_of_words that also
        appears in group.important_words_said, and log(1/3) otherwise.
        Normalised via logsumexp.
        """
        spoken_set = set(group.important_words_said)
        log_p = np.zeros(len(self.slides))
        for s_idx, slide in enumerate(self.slides):
            for w in slide.bag_of_words:
                log_p[s_idx] += self._LOG_IN if w in spoken_set else self._LOG_OUT
        log_p -= logsumexp(log_p)
        return np.exp(log_p)

    def _emit_picture(self, group) -> np.ndarray:
        """
        Inverse L1 pixel distance between group.frame_transformed and each slide image.
        Falls back to uniform when no frame is available.
        """
        n = len(self.slides)
        if group.frame_transformed is None:
            return np.ones(n) / n
        frame_norm = normalize_gray(group.frame_transformed)
        diffs = np.array([np.sum(np.abs(frame_norm - s.slide_img)) for s in self.slides])
        inv = 1.0 / (diffs + 1e-6)
        return inv / inv.sum()

    def _emit_slide_words(self, group) -> np.ndarray:
        """
        Jaccard IoU between words on the group's frame (OCR) and each slide's bag_of_words.
        Normalised so the vector sums to 1.
        """
        n = len(self.slides)
        frame_words = set(group.words_on_slide)
        scores = np.zeros(n)
        for s_idx, slide in enumerate(self.slides):
            union = slide.bag_of_words | frame_words
            inter = slide.bag_of_words & frame_words
            scores[s_idx] = len(inter) / len(union) if union else 0.0
        total = scores.sum()
        if total > 0:
            return scores / total
        return np.ones(n) / n

    def _emit_spoken_iou(self, group) -> np.ndarray:
        """
        Jaccard IoU between group.important_words_said and each slide's bag_of_words.
        Normalised so the vector sums to 1.
        """
        n = len(self.slides)
        spoken = set(group.important_words_said)
        scores = np.zeros(n)
        for s_idx, slide in enumerate(self.slides):
            union = slide.bag_of_words | spoken
            inter = slide.bag_of_words & spoken
            scores[s_idx] = len(inter) / len(union) if union else 0.0
        total = scores.sum()
        if total > 0:
            return scores / total
        return np.ones(n) / n

    def get_log_emission(self, group) -> np.ndarray:
        """
        Multiply the three normalised emission probabilities and return log-normalised result.
        """
        # TODO _emit_spoken_words _emit_spoken_iou perform too bad ( possible fix: parameter for temperature  / rework them), also it should be possible to change dynamically what emissions to use
        joint = (
            # self._emit_spoken_words(group)
             self._emit_picture(group)
            * self._emit_slide_words(group)
            #* self._emit_spoken_iou(group)
        )
        total = joint.sum()
        if total > 0:
            joint /= total
        else:
            joint = np.ones(len(self.slides)) / len(self.slides)
        return np.log(np.clip(joint, 1e-300, None))


class AssignmentModel(EmissionHandler):
    """
    Combines emission computation with HMM forward-backward to assign slides to groups.

    assign_slides(groups) returns [(group, slide_idx)] for each group, 0-based slide index.
    """

    def __init__(self, slides: list, b: float = 0.1):
        super().__init__(slides)
        n = len(slides)
        T = np.full((n, n), (1.0 - b) / (n - 1))
        np.fill_diagonal(T, b)
        self._log_T = np.log(T)

    def _make_log_emit(self, groups: list) -> np.ndarray:
        log_emit = np.zeros((len(groups), len(self.slides)), dtype=np.float64)
        for g_idx, g in enumerate(groups):
            log_emit[g_idx] = self.get_log_emission(g)
        return log_emit

    def _forward(self, log_emit: np.ndarray) -> np.ndarray:
        n_groups, n_slides = log_emit.shape
        log_pi = np.full(n_slides, -np.log(n_slides), dtype=np.float64)
        log_alpha = np.zeros((n_groups, n_slides), dtype=np.float64)
        log_alpha[0] = log_pi + log_emit[0]
        for t in range(1, n_groups):
            trans = log_alpha[t - 1, :, np.newaxis] + self._log_T
            log_alpha[t] = logsumexp(trans, axis=0) + log_emit[t]
        return log_alpha

    def _backward(self, log_emit: np.ndarray) -> np.ndarray:
        n_groups, n_slides = log_emit.shape
        log_beta = np.zeros((n_groups, n_slides), dtype=np.float64)
        for t in range(n_groups - 2, -1, -1):
            contrib = self._log_T + log_emit[t + 1] + log_beta[t + 1]
            log_beta[t] = logsumexp(contrib, axis=1)
        return log_beta

    def assign_slides(self, groups: list) -> list:
        """
        Run the full forward-backward pass and return [(group, slide_idx)] — 0-based.
        """
        log_emit = self._make_log_emit(groups)
        log_alpha = self._forward(log_emit)
        log_beta = self._backward(log_emit)
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        map_slide = np.argmax(np.exp(log_gamma), axis=1)
        return list(zip(groups, map_slide.tolist()))


def export_script(
    assignment: list,
    slides: list,
    groups: list,
    output_path: Path,
) -> None:
    """
    Merge consecutive groups assigned to the same slide, then write a structured
    slide transcript.

    assignment  : [(FrameGroup, slide_idx)] from AssignmentModel.assign_slides
    slides      : list of Slide objects
    groups      : list of FrameGroup (defines ordering — already ordered by FrameReader)
    output_path : destination .txt file
    """
    merged: list[tuple[int, float, float, list[str]]] = []
    for group, slide_idx in assignment:
        if merged and merged[-1][0] == slide_idx:
            s_idx, start, _, texts = merged[-1]
            merged[-1] = (s_idx, start, group.end_time, texts + [group.continuous_text])
        else:
            merged.append((slide_idx, group.start_time, group.end_time, [group.continuous_text]))

    with open(output_path, "w", encoding="utf-8") as f:
        for slide_idx, start, end, texts in merged:
            f.write(
                f"slide #{slide_idx + 1}  "
                f"[{format_timestamp(start)} --> {format_timestamp(end)}]\n"
            )
            f.write("  " + " ".join(texts).strip() + "\n\n")
