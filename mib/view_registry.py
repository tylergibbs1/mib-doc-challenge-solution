"""Best-effort, append-only identities for pixels already used by inference.

The registry is diagnostic only.  It never renders, decodes, rotates, or
otherwise constructs a decision-bearing image; callers hand it the exact array
that already produced an OCR or decoder result.  Every public operation is
fail-soft so provenance cannot change adjudication.
"""
import copy
import hashlib
import math
import re

import numpy as np


SCHEMA = "mib-image-view-registry-v1"
_TOKEN_RE = re.compile(r"[a-z][a-z0-9_]*")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def empty_snapshot(error_type=None):
    """Return one valid empty registry, optionally noting snapshot failure."""
    errors = [] if error_type is None else [{
        "page": None,
        "semantic_id": "snapshot",
        "error_type": str(error_type)[:80],
    }]
    return {"schema": SCHEMA, "pages": [], "errors": errors}


class ImageViewRegistry:
    """A private append-only event sink with deterministic serialization."""

    def __init__(self):
        self._events = {}
        self._semantic_ids = set()
        self._errors = []

    @staticmethod
    def _token(value, name):
        if not isinstance(value, str) or not _TOKEN_RE.fullmatch(value):
            raise ValueError(f"invalid_{name}")
        return value

    @staticmethod
    def _page(value):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("invalid_page")
        return value

    @staticmethod
    def _finite_number(value, name, positive=False):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"invalid_{name}")
        number = float(value)
        if not math.isfinite(number) or (positive and number <= 0):
            raise ValueError(f"invalid_{name}")
        return number

    @staticmethod
    def _shape(value):
        if (not isinstance(value, (list, tuple)) or len(value) != 2
                or any(isinstance(item, bool) or not isinstance(item, int)
                       or item <= 0 for item in value)):
            raise ValueError("invalid_shape")
        return [int(item) for item in value]

    def _error(self, page, semantic_id, error_type):
        try:
            safe_page = page if isinstance(page, int) and not isinstance(
                page, bool) and page >= 0 else None
            self._errors.append({
                "page": safe_page,
                "semantic_id": str(semantic_id)[:240],
                "error_type": str(error_type)[:80],
            })
        except Exception:
            pass

    def observe_fingerprint(self, *, page, consumer, pass_name, transform,
                            source, dpi, rotation_degrees, shape, dtype,
                            pixel_sha256, preprocess="none"):
        """Append one frozen fingerprint, rejecting semantic duplicates."""
        semantic_id = ":".join(str(value) for value in (
            page, consumer, pass_name, transform))
        try:
            page = self._page(page)
            consumer = self._token(consumer, "consumer")
            pass_name = self._token(pass_name, "pass")
            transform = self._token(transform, "transform")
            source = self._token(source, "source")
            preprocess = self._token(preprocess, "preprocess")
            dpi = self._finite_number(dpi, "dpi", positive=True)
            rotation = self._finite_number(
                rotation_degrees, "rotation_degrees")
            shape = self._shape(shape)
            if dtype != "uint8":
                raise ValueError("invalid_dtype")
            if (not isinstance(pixel_sha256, str)
                    or not _SHA256_RE.fullmatch(pixel_sha256)):
                raise ValueError("invalid_pixel_sha256")
            semantic = (page, consumer, pass_name, transform)
            if semantic in self._semantic_ids:
                self._error(page, semantic_id, "duplicate_semantic_identity")
                return False
            events = self._events.setdefault(page, [])
            ordinal = len(events)
            record = {
                "view_id": (f"p{page}:{ordinal}:{consumer}:{pass_name}:"
                            f"{transform}"),
                "page": page,
                "ordinal": ordinal,
                "consumer": consumer,
                "pass": pass_name,
                "transform": transform,
                "source": source,
                "preprocess": preprocess,
                "dpi": dpi,
                "rotation_degrees": rotation,
                "shape": shape,
                "dtype": dtype,
                "pixel_sha256": pixel_sha256,
            }
            self._semantic_ids.add(semantic)
            events.append(record)
            return True
        except Exception as exc:
            self._error(page, semantic_id, type(exc).__name__)
            return False

    def observe_pixels(self, *, image, **identity):
        """Fingerprint one exact two-dimensional uint8 inference array."""
        page = identity.get("page")
        semantic_id = ":".join(str(identity.get(key)) for key in (
            "page", "consumer", "pass_name", "transform"))
        try:
            if not isinstance(image, np.ndarray) or image.ndim != 2 \
                    or image.dtype != np.uint8 or not image.size:
                raise ValueError("invalid_image")
            contiguous = np.ascontiguousarray(image)
            return self.observe_fingerprint(
                **identity,
                shape=[int(value) for value in contiguous.shape],
                dtype=str(contiguous.dtype),
                pixel_sha256=hashlib.sha256(
                    contiguous.tobytes()).hexdigest())
        except Exception as exc:
            self._error(page, semantic_id, type(exc).__name__)
            return False

    def snapshot(self):
        """Return detached page-ascending, page-local-append-order records."""
        try:
            pages = [{
                "page": page,
                "events": copy.deepcopy(self._events[page]),
            } for page in sorted(self._events)]
            return {
                "schema": SCHEMA,
                "pages": pages,
                "errors": copy.deepcopy(self._errors),
            }
        except Exception as exc:
            return empty_snapshot(type(exc).__name__)
