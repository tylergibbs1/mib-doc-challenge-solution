"""Exact lazy adapter for the pre-M0 OCR implementation."""

from __future__ import annotations

from typing import Any


def recognize(image: Any, *, min_lines: int = 4, hq: bool = False):
    from . import ocr

    return ocr.ocr_page(image, min_lines=min_lines, hq=hq)


__all__ = ["recognize"]
