"""Unavailable PP-OCRv6 placeholder with no M0 runtime side effects."""


class BackendUnavailable(RuntimeError):
    """The PP-OCRv6 backend has not been materialized and pinned."""


def create_backend():
    raise BackendUnavailable("PP-OCRv6 backend is unavailable in M0")


__all__ = ["BackendUnavailable", "create_backend"]
