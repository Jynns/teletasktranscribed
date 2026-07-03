import cv2
import numpy as np
from scipy.special import logsumexp

from util import extract_interesting_words, ocr_crop, normalize_gray


class Slide:
    """
    Represents one PDF slide.

    init args:
        image  : ndarray (BGR or grayscale) — already rendered, not a path.
        text   : optional pre-extracted string (e.g. from fitz).
                 When provided, used instead of OCR for bag_of_words.
        target_size : (W, H) for the visual embedding.
    """

    def __init__(self, image, text: str = None, target_size: tuple = (128, 128)):
        # Extract words: pre-extracted text wins over OCR
        if text is not None:
            self.bag_of_words: set = set(extract_interesting_words(text))
        else:
            self.bag_of_words: set = set(ocr_crop(image))

        # Grayscale + resize
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_LANCZOS4)

        self._slide_img_uint8 = resized
        self.slide_img = normalize_gray(resized)  # float32, z-score — used for visual emission

    @property
    def slide_img_uint8(self) -> np.ndarray:
        """uint8 grayscale at target_size — for visualisation."""
        return self._slide_img_uint8


class SlideHandler:
    """
    Receives pre-rendered slide images and wraps them as Slide instances.

    slide_data : list of ndarray  (just images)
                 OR list of (image, text_or_None) tuples.
    """

    def __init__(self, slide_data, target_size: tuple = (128, 128)):
        self.slides: list[Slide] = []
        for item in slide_data:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                image, text = item
            else:
                image, text = item, None
            self.slides.append(Slide(image, text=text, target_size=target_size))


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
        Normalise via logsumexp and return as probabilities.
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
        Compute total absolute pixel difference between the group's normalised frame
        and each slide's normalised image.  Weights are inverse distances, normalised
        to sum to 1.  Falls back to uniform when no frame is available.
        """
        n = len(self.slides)
        if group.frame is None:
            return np.ones(n) / n
        frame_norm = normalize_gray(group.frame)
        diffs = np.array([np.sum(np.abs(frame_norm - s.slide_img)) for s in self.slides])
        inv = 1.0 / (diffs + 1e-6)
        return inv / inv.sum()

    def _emit_slide_words(self, group) -> np.ndarray:
        """
        Jaccard similarity between words recognised on the group's frame (OCR) and
        each slide's bag_of_words.  Normalised so the vector sums to 1.
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

    def get_log_emission(self, group) -> np.ndarray:
        """
        Multiply the three normalised emission probabilities and return the
        log-normalised joint probability vector over all slides.
        """
        joint = (
            self._emit_spoken_words(group)
            * self._emit_picture(group)
            * self._emit_slide_words(group)
        )
        total = joint.sum()
        if total > 0:
            joint /= total
        else:
            joint = np.ones(len(self.slides)) / len(self.slides)
        return np.log(np.clip(joint, 1e-300, None))
