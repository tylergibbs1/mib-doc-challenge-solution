"""Capability-gated OCR experiment boundary.

This module is an accidental-misuse boundary, not a hostile Python sandbox.
The default production pipeline does not import or call it in M0.
"""

from __future__ import annotations

import hashlib
import math
import weakref
from dataclasses import dataclass
from typing import Any

import numpy as np


SAFE_SOURCES = frozenset({
    "masked_pdf_render",
    "composited_pdf_render",
    "native_full_page_image",
    "p0b_masked_scan_image",
})


class OcrBoundaryError(RuntimeError):
    """Base class for rejected experimental OCR operations."""


class InvalidOcrInput(OcrBoundaryError):
    """The proposed OCR input was not an exact authorized raster."""


class InputMutationError(OcrBoundaryError):
    """The legacy input changed after the safe snapshot was issued."""


class CandidateOutputError(OcrBoundaryError):
    """A candidate returned something other than strict OCR pairs."""


_VIEW_TOKEN = object()
_HANDLE_TOKEN = object()


class SanitizedImageView:
    """Narrow pixels-only input exposed to a candidate backend."""

    __slots__ = ("__pixels", "__source")

    def __init__(self, token: object, pixels: np.ndarray, source: str) -> None:
        if token is not _VIEW_TOKEN:
            raise TypeError("SanitizedImageView is issued by the trusted bridge")
        self.__pixels = pixels
        self.__source = source

    @property
    def pixels(self) -> np.ndarray:
        return self.__pixels

    @property
    def shape(self) -> tuple[int, int]:
        return self.__pixels.shape

    @property
    def source(self) -> str:
        return self.__source


class RegisteredOcrInput:
    """Opaque one-shot handle whose state remains outside the object."""

    __slots__ = ("__weakref__",)

    def __new__(cls, token: object):
        if cls is not RegisteredOcrInput or token is not _HANDLE_TOKEN:
            raise TypeError("RegisteredOcrInput is issued by the trusted bridge")
        return super().__new__(cls)

    def __init__(self, token: object) -> None:
        del token


@dataclass(frozen=True, slots=True)
class _OriginalSeal:
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: str
    writeable: bool
    digest: bytes


@dataclass(frozen=True, slots=True)
class _InputState:
    original: np.ndarray
    seal: _OriginalSeal
    view: SanitizedImageView


_ISSUED: weakref.WeakSet[RegisteredOcrInput] = weakref.WeakSet()
_STATES: weakref.WeakKeyDictionary[RegisteredOcrInput, _InputState] = (
    weakref.WeakKeyDictionary()
)


def _validate_original(image: Any, source: Any) -> np.ndarray:
    if type(image) is not np.ndarray:
        raise InvalidOcrInput("image must be an exact numpy.ndarray")
    if image.ndim != 2 or image.size == 0 or image.dtype != np.uint8:
        raise InvalidOcrInput(
            "image must be a nonempty two-dimensional uint8 array"
        )
    if type(source) is not str or source not in SAFE_SOURCES:
        raise InvalidOcrInput("source is not an exact authorized raster source")
    return image


def _seal(image: np.ndarray) -> _OriginalSeal:
    payload = image.tobytes(order="C")
    return _OriginalSeal(
        shape=tuple(int(value) for value in image.shape),
        strides=tuple(int(value) for value in image.strides),
        dtype=image.dtype.str,
        writeable=bool(image.flags.writeable),
        digest=hashlib.sha256(payload).digest(),
    )


def _snapshot(image: np.ndarray, source: str) -> SanitizedImageView:
    # A bytes root prevents setflags(write=True) from making the snapshot
    # mutable, while C-order serialization detaches every unusual input stride.
    owned_bytes = image.tobytes(order="C")
    pixels = np.frombuffer(owned_bytes, dtype=np.uint8).reshape(image.shape)
    if pixels.flags.writeable or not pixels.flags.c_contiguous:
        raise AssertionError("internal sanitized snapshot invariant failed")
    return SanitizedImageView(_VIEW_TOKEN, pixels, source)


def _issue_registered_input(*, image: Any, source: Any) -> RegisteredOcrInput:
    original = _validate_original(image, source)
    before = _seal(original)
    view = _snapshot(original, source)
    after = _seal(original)
    if before != after:
        raise InputMutationError("OCR input changed while its snapshot was issued")

    handle = RegisteredOcrInput(_HANDLE_TOKEN)
    _STATES[handle] = _InputState(original=original, seal=after, view=view)
    _ISSUED.add(handle)
    return handle


def _trusted_forensics_issuer():
    """Return a capability consumed by its first issuance attempt.

    Trusted forensics/pipeline code obtains a new private callable per view.
    The leading underscore and one-shot closure are review guardrails; Python
    module privacy is not a security boundary.
    """

    used = False

    def issue(*, image: Any, source: Any) -> RegisteredOcrInput:
        nonlocal used
        if used:
            raise OcrBoundaryError("trusted OCR issuer capability already consumed")
        used = True
        return _issue_registered_input(image=image, source=source)

    return issue


def _assert_unchanged(original: np.ndarray, expected: _OriginalSeal) -> None:
    try:
        actual = _seal(original)
    except Exception as exc:
        raise InputMutationError("legacy OCR input became invalid") from exc
    if actual != expected:
        raise InputMutationError("legacy OCR input changed after issue")


def _consume(
    handle: Any,
) -> tuple[np.ndarray, _OriginalSeal, SanitizedImageView]:
    if type(handle) is not RegisteredOcrInput or handle not in _ISSUED:
        raise InvalidOcrInput("OCR input handle was not issued or was consumed")

    # Revoke authority before any fallible validation or backend import.
    _ISSUED.discard(handle)
    try:
        state = _STATES.pop(handle)
    except KeyError as exc:
        raise InvalidOcrInput("OCR input handle has no registered state") from exc
    _assert_unchanged(state.original, state.seal)
    return state.original, state.seal, state.view


def _normalise_candidate_output(value: Any) -> list[tuple[str, float]]:
    if type(value) not in (list, tuple):
        raise CandidateOutputError("candidate output must be an exact list or tuple")

    normalised = []
    for pair in value:
        if type(pair) not in (list, tuple) or len(pair) != 2:
            raise CandidateOutputError(
                "each OCR item must be an exact two-item list or tuple"
            )
        text, confidence = pair
        if type(text) is not str or not text.strip():
            raise CandidateOutputError("OCR text must be an exact nonblank string")
        if type(confidence) not in (int, float):
            raise CandidateOutputError(
                "OCR confidence must be an exact built-in int or float"
            )
        numeric_confidence = float(confidence)
        if (
            not math.isfinite(numeric_confidence)
            or not 0.0 <= numeric_confidence <= 1.0
        ):
            raise CandidateOutputError(
                "OCR confidence must be finite and in [0, 1]"
            )
        normalised.append((text, numeric_confidence))
    return normalised


def _select_candidate_factory(backend: Any):
    if type(backend) is not str:
        raise OcrBoundaryError("candidate backend name must be an exact string")
    if backend == "v6":
        from .ocr_v6 import create_backend

        return create_backend
    raise OcrBoundaryError(f"candidate OCR backend is unavailable: {backend!r}")


def _legacy(original: np.ndarray, *, min_lines: int, hq: bool):
    from . import ocr_legacy

    return ocr_legacy.recognize(original, min_lines=min_lines, hq=hq)


def recognize(
    handle: Any,
    backend: str = "legacy",
    *,
    min_lines: int = 4,
    hq: bool = False,
):
    """Use exact legacy OCR or try one candidate with safe legacy fallback."""

    original, seal, view = _consume(handle)
    if type(backend) is str and backend == "legacy":
        return _legacy(original, min_lines=min_lines, hq=hq)

    candidate_failed = False
    candidate_result = None
    try:
        factory = _select_candidate_factory(backend)
        candidate = factory()
        raw_result = candidate.recognize(view, min_lines=min_lines, hq=hq)
        candidate_result = _normalise_candidate_output(raw_result)
    except Exception:
        candidate_failed = True

    # These checks deliberately sit outside the candidate exception handler:
    # mutation is never converted into fallback, and a later legacy exception
    # has no candidate exception as its active context.
    _assert_unchanged(original, seal)
    if candidate_failed:
        return _legacy(original, min_lines=min_lines, hq=hq)
    return candidate_result


__all__ = [
    "CandidateOutputError",
    "InputMutationError",
    "InvalidOcrInput",
    "OcrBoundaryError",
    "RegisteredOcrInput",
    "SAFE_SOURCES",
    "SanitizedImageView",
    "recognize",
]
