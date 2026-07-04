import re
import cv2
import numpy as np

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


class Timestamp:
    def __init__(self, start, end):
        self.start = start
        self.end = end


STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'is',
    'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
    'did', 'will', 'would', 'shall', 'should', 'may', 'might', 'can', 'could',
    'not', 'no', 'nor', 'so', 'yet', 'both', 'either', 'neither', 'just', 'also',
    'that', 'this', 'these', 'those', 'it', 'its', 'we', 'our', 'you', 'your',
    'he', 'she', 'they', 'them', 'then', 'than', 'with', 'from', 'into', 'through',
    'about', 'what', 'which', 'who', 'how', 'when', 'where', 'why', 'if', 'as',
    'by', 'up', 'out', 'over', 'some', 'all', 'each', 'every', 'more', 'other',
    'there', 'their', 'here',
}

MIN_WORD_LEN = 3


def tokenize(text: str) -> list:
    """Lowercase, strip punctuation, split, remove stopwords and short tokens."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return [w for w in text.split() if len(w) >= MIN_WORD_LEN and w not in STOPWORDS]


def extract_interesting_words(text: str) -> list:
    return tokenize(text)


def frame_diff(f1, f2) -> float:
    """Mean absolute pixel difference between two uint8 frames. None -> inf."""
    if f1 is None or f2 is None:
        return float('inf')
    return float(np.mean(np.abs(f1.astype(np.int32) - f2.astype(np.int32))))


def normalize_gray(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    s = a.std()
    return (a - a.mean()) / (s if s > 0 else 1.0)


def ocr_crop(roi_bgr) -> list:
    """Extract and tokenize words visible in a cropped BGR or grayscale frame."""
    if not TESSERACT_AVAILABLE:
        return []
    if len(roi_bgr.shape) == 3:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_bgr
    h, w = gray.shape
    scale = max(2, 600 // max(w, 1))
    big = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(thresh, lang='eng', config='--psm 6')
    return tokenize(text)


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def find_aoi_corner(bgr, ksize=3, threshold=80, angle_tol=6):
    """Return (x_crop, y_crop) of the slide AOI upper-left corner."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    sx  = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=ksize)
    sy  = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=ksize)
    sx2 = cv2.Sobel(np.abs(sx), cv2.CV_64F, 1, 0, ksize=ksize)
    sy2 = cv2.Sobel(np.abs(sy), cv2.CV_64F, 0, 1, ksize=ksize)
    bx = cv2.threshold(cv2.convertScaleAbs(sx + 3*sx2), threshold, 255, cv2.THRESH_BINARY)[1]
    by = cv2.threshold(cv2.convertScaleAbs(sy + 3*sy2), threshold, 255, cv2.THRESH_BINARY)[1]
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    closed = cv2.bitwise_or(cv2.morphologyEx(bx, cv2.MORPH_CLOSE, kv),
                             cv2.morphologyEx(by, cv2.MORPH_CLOSE, kh))
    lines = cv2.HoughLinesP(closed, 1, np.pi/180, 40, minLineLength=40, maxLineGap=50)
    h_lines, v_lines = [], []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            length = np.hypot(x2-x1, y2-y1)
            if abs(y2-y1) <= angle_tol:   h_lines.append((length, l[0]))
            elif abs(x2-x1) <= angle_tol: v_lines.append((length, l[0]))
    h_lines.sort(key=lambda t: -t[0])
    v_lines.sort(key=lambda t: -t[0])
    cx = int((v_lines[0][1][0]+v_lines[0][1][2])/2) if v_lines else None
    cy = int((h_lines[0][1][1]+h_lines[0][1][3])/2) if h_lines else None
    return cx, cy
