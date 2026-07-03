import cv2
import numpy as np
from scipy.special import logsumexp

from util import extract_interesting_words, ocr_crop, normalize_gray


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

    def get_log_emission(self, group) -> np.ndarray:
        """
        Multiply the three normalised emission probabilities and return log-normalised result.
        """
        joint = (
            # self._emit_spoken_words(group)
             self._emit_picture(group)
            * self._emit_slide_words(group)
        )
        total = joint.sum()
        if total > 0:
            joint /= total
        else:
            joint = np.ones(len(self.slides)) / len(self.slides)
        return np.log(np.clip(joint, 1e-300, None))
