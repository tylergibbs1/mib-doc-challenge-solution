from __future__ import annotations

import io
import os
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pymupdf

FOOTER_RE = re.compile(
    r"(?:Packet\s+MIB-\d{6}\s*/\s*page\s*\d+|Synthetic hiring challenge document)",
    re.IGNORECASE,
)


@dataclass
class PageEvidence:
    page_number: int
    document_type: str
    text: str
    native_text: str
    ocr_text: str
    ocr_mean_confidence: float
    ocr_used: bool
    visible_native_characters: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> PageEvidence:
        return cls(**value)


def _rgb(color: int) -> tuple[int, int, int]:
    return ((color >> 16) & 255, (color >> 8) & 255, color & 255)


def _is_visible_span(span: dict, crop: pymupdf.Rect) -> bool:
    text = str(span.get("text", "")).strip()
    if not text or float(span.get("size", 0.0)) < 4.5:
        return False
    bbox = pymupdf.Rect(span.get("bbox", (0, 0, 0, 0)))
    intersection = bbox & crop
    if bbox.is_empty or intersection.is_empty:
        return False
    if intersection.get_area() < 0.75 * max(1.0, bbox.get_area()):
        return False
    alpha = int(span.get("alpha", 255))
    red, green, blue = _rgb(int(span.get("color", 0)))
    # Challenge injections are commonly opaque white text on a white page.
    # Conservative luminance filtering also rejects very faint decoys.
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    contrast = (255.0 - luminance) * (alpha / 255.0)
    return contrast >= 28.0


def visible_native_text(page: pymupdf.Page) -> str:
    lines: list[str] = []
    crop = page.rect
    payload = page.get_text("dict", sort=True)
    for block in payload.get("blocks", []):
        for line in block.get("lines", []):
            parts = [
                str(span.get("text", "")).strip()
                for span in line.get("spans", [])
                if _is_visible_span(span, crop)
            ]
            text = " ".join(part for part in parts if part).strip()
            if text:
                lines.append(text)
    return "\n".join(lines)


def _preprocess(gray: np.ndarray) -> np.ndarray:
    low, high = np.percentile(gray, (1.0, 99.0))
    if high - low < 80:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12)).apply(gray)
    return gray


def _remove_form_lines(gray: np.ndarray) -> np.ndarray:
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        13,
    )
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(24, gray.shape[1] // 28), 1)
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(24, gray.shape[0] // 35))
    )
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    cleaned = cv2.subtract(binary, cv2.bitwise_or(horizontal, vertical))
    cleaned = cv2.morphologyEx(
        cleaned, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    )
    return cv2.bitwise_not(cleaned)


def _ocr_quality(text: str, confidence: float) -> float:
    normalized = re.sub(r"[^A-Z0-9]+", " ", text.upper())
    anchors = (
        "MIB",
        "FORM",
        "APPLICANT",
        "SPECIES",
        "SPONSOR",
        "SPN",
        "ARRIVAL",
        "FEE",
        "FEE STATUS",
        "PAID",
        "WAIVED",
        "UNPAID",
        "REGISTRY",
        "BIOMETRIC",
        "FINDING",
        "DENIED",
        "APPROVED",
        "NEEDS REVIEW",
    )
    anchor_score = 12.0 * sum(anchor in normalized for anchor in anchors)
    return confidence + min(18.0, len(re.sub(r"\W", "", text)) / 30.0) + anchor_score


def _tsv_to_text(tsv: str) -> tuple[str, float]:
    groups: dict[tuple[str, str, str, str], list[tuple[int, str]]] = {}
    confidences: list[float] = []
    # Tesseract emits TSV, not RFC-compliant quoted CSV. In particular, a
    # recognized `"` glyph is not escaped; csv.DictReader can then consume
    # subsequent coordinate rows as one giant text field. Split a maximum of
    # eleven times so the final OCR-text column is preserved verbatim.
    for raw_line in io.StringIO(tsv):
        parts = raw_line.rstrip("\r\n").split("\t", 11)
        if len(parts) != 12 or parts[0] == "level":
            continue
        word = parts[11].strip()
        if not word:
            continue
        try:
            confidence = float(parts[10])
        except ValueError:
            confidence = -1.0
        if confidence >= 0:
            confidences.append(confidence)
        key = (parts[1], parts[2], parts[3], parts[4])
        try:
            left = int(parts[6])
        except ValueError:
            left = 0
        groups.setdefault(key, []).append((left, word))
    lines = [" ".join(word for _, word in sorted(words)) for words in groups.values()]
    mean_confidence = float(np.mean(confidences)) if confidences else 0.0
    return "\n".join(lines), mean_confidence


def _ocr(gray: np.ndarray, psm: int = 11) -> tuple[str, float]:
    ok, encoded = cv2.imencode(".png", gray)
    if not ok:
        return "", 0.0
    env = os.environ.copy()
    env.setdefault("OMP_THREAD_LIMIT", "1")
    command = [
        "tesseract",
        "stdin",
        "stdout",
        "--dpi",
        "220",
        "--psm",
        str(psm),
        "-l",
        "eng",
        "-c",
        "preserve_interword_spaces=1",
        "tsv",
    ]
    result = subprocess.run(
        command,
        input=encoded.tobytes(),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        env=env,
        timeout=25,
    )
    if result.returncode != 0:
        return "", 0.0
    return _tsv_to_text(result.stdout.decode("utf-8", errors="replace"))


def render_and_ocr(page: pymupdf.Page, dpi: int = 220) -> tuple[str, float]:
    pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False, annots=True)
    gray = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width)
    text, confidence = _ocr(gray, psm=11)
    best = (text, confidence)
    # Targeted retries help faded pages and forms whose rules intersect glyphs,
    # without paying a 2-3x OCR cost on ordinary pages.
    if confidence < 70.0 or len(re.sub(r"\W", "", text)) < 70:
        for variant in (_preprocess(gray), _remove_form_lines(gray)):
            retry_text, retry_confidence = _ocr(variant, psm=11)
            if _ocr_quality(retry_text, retry_confidence) > _ocr_quality(*best):
                best = (retry_text, retry_confidence)
    return best


def poppler_ocr(pdf_path: Path, page_number: int, dpi: int = 220) -> tuple[str, float]:
    with tempfile.TemporaryDirectory(prefix="mib-poppler-") as directory:
        prefix = Path(directory) / "page"
        command = [
            "pdftoppm",
            "-f",
            str(page_number),
            "-l",
            str(page_number),
            "-singlefile",
            "-gray",
            "-r",
            str(dpi),
            "-q",
            str(pdf_path),
            str(prefix),
        ]
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=20,
        )
        image_path = prefix.with_suffix(".pgm")
        if result.returncode != 0 or not image_path.exists():
            return "", 0.0
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return "", 0.0
        text, confidence = _ocr(gray, psm=11)
        normalized = re.sub(r"[^A-Z0-9]+", " ", text.upper())
        if "MANUAL" in normalized or "ADJUDICATOR" in normalized:
            layout_text, layout_confidence = _ocr(gray, psm=3)
            if layout_text and layout_text != text:
                text = f"{text}\n{layout_text}"
                confidence = max(confidence, layout_confidence)
        return text, confidence


def classify_document(text: str) -> str:
    normalized = re.sub(r"[^A-Z0-9]+", " ", text.upper())
    if "MANUAL ADJUDICATOR" in normalized or "FINDING" in normalized:
        return "manual"
    if (
        "WORK AUTHORIZATION" in normalized
        or "PRIMARY INTAKE" in normalized
        or "FORM I 8090" in normalized
        or "FORM J 8090" in normalized
    ):
        return "intake"
    if "BIOMETRIC" in normalized or "BIOHAZARD" in normalized:
        return "biometric"
    if "FEE RECEIPT" in normalized or ("FEE STATUS" in normalized and "AMOUNT" in normalized):
        return "fee"
    if "SPONSOR ATTESTATION" in normalized or "SPONSOR" in normalized and "ATTESTS" in normalized:
        return "sponsor"
    if "REGISTRY EXTRACT" in normalized or "REGISTRY STATUS" in normalized:
        return "registry"
    return "unknown"


def extract_pdf(pdf_path: str | Path, dpi: int = 220) -> dict:
    pdf_path = Path(pdf_path)
    pages: list[PageEvidence] = []
    with pymupdf.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            native = visible_native_text(page)
            try:
                ocr_text, ocr_confidence = render_and_ocr(page, dpi=dpi)
                if ocr_confidence < 78.0 or len(re.sub(r"\W", "", ocr_text)) < 80:
                    alternate_text, alternate_confidence = poppler_ocr(
                        pdf_path, page_index + 1, dpi=dpi
                    )
                    if _ocr_quality(alternate_text, alternate_confidence) > _ocr_quality(
                        ocr_text, ocr_confidence
                    ):
                        ocr_text, ocr_confidence = alternate_text, alternate_confidence
            except (subprocess.SubprocessError, ValueError, RuntimeError):
                ocr_text, ocr_confidence = "", 0.0
            # Sanitized native text is exact on digital pages. Put it first so
            # equal-priority evidence does not prefer a noisier OCR duplicate.
            merged = "\n".join(part for part in (native, ocr_text) if part)
            pages.append(
                PageEvidence(
                    page_number=page_index + 1,
                    document_type=classify_document(merged),
                    text=merged,
                    native_text=native,
                    ocr_text=ocr_text,
                    ocr_mean_confidence=ocr_confidence,
                    ocr_used=bool(ocr_text),
                    visible_native_characters=len(FOOTER_RE.sub("", native).strip()),
                )
            )
    return {
        "case_id": pdf_path.stem,
        "pdf_name": pdf_path.name,
        "pages": [page.to_dict() for page in pages],
    }


def safe_extract_pdf(pdf_path: str | Path, dpi: int = 220) -> dict:
    pdf_path = Path(pdf_path)
    try:
        return extract_pdf(pdf_path, dpi=dpi)
    except Exception as error:  # noqa: BLE001 - case isolation is intentional
        # One corrupt packet must not discard predictions already written for
        # thousands of other cases. Empty evidence resolves to valid unknown
        # fields and a conservative decision.
        return {
            "case_id": pdf_path.stem,
            "pdf_name": pdf_path.name,
            "pages": [],
            "extraction_error": f"{type(error).__name__}: {error}",
        }
