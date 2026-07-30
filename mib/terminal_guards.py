"""Evidence-only terminal-to-review guards.

The detector records a compact summary of visible structure on each physical
page's primary view.  It deliberately excludes HQ duplicates and contains no
case-specific membership, labels, or hidden-text inputs.
"""
from __future__ import annotations

import re

from .vocab import VISAS


OBSERVATION_SCHEMA = "mib-terminal-page-observation-v1"
LOCK_SCHEMA = "mib-terminal-guard-lock-v1"

G1 = "g1_missing_visa_conditional_denial"
G2 = "g2_unresolved_exact_identity_conflict"
G3 = "g3_superseded_benign_visa_approval"

CONDITIONAL_REASONS = frozenset(
    {"embargoed_world", "revoked_sponsor", "stale_application"}
)
SOFT_WORLDS = frozenset({"Wolf-1061c"})
ADVERSE_VISAS = frozenset({"TRANSIT-7"})

FOOTER_RE = re.compile(
    r"^\s*packet\s+(MIB-\d{6})\s*/\s*page\s+\d+\s*$",
    re.IGNORECASE,
)
BODY_CASE_RE = re.compile(
    r"^\s*case\s*id\s*[:.]\s*(MIB-\d{6})\s*$",
    re.IGNORECASE,
)
TRIAD_RES = (
    re.compile(r"\bCOPY\b", re.IGNORECASE),
    re.compile(r"\bFILED\b", re.IGNORECASE),
    re.compile(r"\bARCHIVE(?:D)?\b", re.IGNORECASE),
)
ARCHIVED_ADJACENT_RE = re.compile(
    r"\b(?:archived\s+adjacent|adjacent\s+applicant)\b",
    re.IGNORECASE,
)
VISA_DAMAGE_RE = re.compile(
    r"(?i)^\s*(?:visa\s*(?:class|type)|v[i1l]sa\s*c[l1i]ass)"
    r"\s*[:.]\s*(.*)$"
)
ABSENT_RE = re.compile(
    r"(?i)\[?\s*[A-Z ]*?(washed\s*out|unreadable|cut\s*out|blank|redacted|"
    r"missing|torn|whiteout|white\s*out|obscured|illegible|lost)\s*\]?\s*$"
)
CASE_ID_RE = re.compile(r"MIB-\d{6}")


def _levenshtein(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, left_character in enumerate(left, 1):
        current = [i]
        for j, right_character in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (
                        left_character != right_character
                    ),
                )
            )
        previous = current
    return previous[-1]


def _sample_token_like(token: str) -> bool:
    if not 4 <= len(token) <= 8:
        return False
    if _levenshtein(token, "SAMPLE") <= 1:
        return True
    # A damaged stamp may lose its left edge.  This is accepted only beside a
    # separately validated DENIAL token.
    return (
        len(token) >= 4
        and token.endswith("PLE")
        and _levenshtein(token, "SAMPLE") <= 3
    )


def _denial_token_like(token: str) -> bool:
    if token == "DENIAL":
        return True
    if len(token) == 5 and any(
        "DENIAL"[:index] + "DENIAL"[index + 1:] == token
        for index in range(len("DENIAL"))
    ):
        return True
    if len(token) != 6:
        return False
    mismatches = [
        (index, expected, observed)
        for index, (expected, observed) in enumerate(zip("DENIAL", token))
        if expected != observed
    ]
    if len(mismatches) != 1:
        return False
    index, expected, observed = mismatches[0]
    return (index, expected, observed) in {
        (2, "N", "H"),
        (3, "I", "L"),
        (5, "L", "I"),
    }


def isolated_sample_denial(line: str) -> bool:
    """Recognize only a short, mostly-uppercase SAMPLE DENIAL stamp line."""
    text = str(line).strip()
    if not text or ":" in text or len(text) > 30:
        return False
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return False
    uppercase_ratio = (
        sum(character.isupper() for character in letters) / len(letters)
    )
    if uppercase_ratio < 0.8:
        return False
    compact = "".join(
        {"1": "I", "|": "I", "!": "I"}.get(character, character)
        for character in text.upper()
        if character.isalnum() or character in "1|!"
    )
    if not 9 <= len(compact) <= 15:
        return False
    return any(
        _sample_token_like(compact[:split])
        and _denial_token_like(compact[split:])
        for split in range(4, min(9, len(compact) - 4))
    )


def _page_lines(page: dict) -> list[str]:
    lines = []
    for item in page.get("lines") or ():
        if isinstance(item, (list, tuple)):
            if item:
                lines.append(str(item[0]))
        else:
            lines.append(str(item))
    return lines


def _page_has_triad(lines: list[str]) -> bool:
    joined = " ".join(lines)
    return all(pattern.search(joined) for pattern in TRIAD_RES)


def _visa_token_pattern(visa: str) -> re.Pattern:
    pieces = [re.escape(piece) for piece in str(visa).upper().split("-")]
    return re.compile(
        r"(?<![A-Z0-9])"
        + r"[\s-]*".join(pieces)
        + r"(?![A-Z0-9])",
        re.IGNORECASE,
    )


def observe_primary_pages(case_id: str, pages) -> dict:
    """Summarize visible primary-page facts needed by the terminal guards."""
    active = str(case_id).upper()
    if not CASE_ID_RE.fullmatch(active):
        raise ValueError("terminal observation requires a canonical case id")

    identity_conflicts = []
    visa_damage = []
    visa_support = {}
    patterns = {
        visa: _visa_token_pattern(visa)
        for visa in VISAS
    }

    for page in pages or ():
        if page.get("kind") not in {"scan", "native"}:
            continue
        lines = _page_lines(page)
        page_number = page.get("page")
        kind = page.get("kind")
        full_triad = _page_has_triad(lines)
        if str(page.get("skipped") or "").lower() in {
            "foreign", "unsafe",
        }:
            # Quarantined pages cannot establish active-case damage or value
            # support. They remain visible to neither terminal direction.
            continue

        for line in lines:
            match = VISA_DAMAGE_RE.fullmatch(line)
            if match and ABSENT_RE.fullmatch(match.group(1).strip()):
                visa_damage.append(
                    {"page": page_number, "kind": kind, "line": line}
                )

        for visa, pattern in patterns.items():
            matching_lines = [
                line for line in lines if pattern.search(line)
            ]
            if matching_lines:
                visa_support.setdefault(visa, []).append(
                    {
                        "page": page_number,
                        "kind": kind,
                        "matching_lines": matching_lines,
                        "full_supersession_triad": full_triad,
                    }
                )

        footer_ids = [
            match.group(1).upper()
            for line in lines
            if (match := FOOTER_RE.fullmatch(line))
        ]
        if active not in footer_ids:
            continue
        body_ids = sorted(
            {
                match.group(1).upper()
                for line in lines
                if (match := BODY_CASE_RE.fullmatch(line))
                and match.group(1).upper() != active
            }
        )
        if not body_ids:
            continue

        sample_lines = [
            line for line in lines if isolated_sample_denial(line)
        ]
        archived_adjacent = bool(
            ARCHIVED_ADJACENT_RE.search(" ".join(lines))
        )
        exclusions = []
        if sample_lines:
            exclusions.append("isolated_sample_denial")
        if full_triad:
            exclusions.append("full_supersession_triad")
        if archived_adjacent:
            exclusions.append("archived_adjacent")
        identity_conflicts.append(
            {
                "page": page_number,
                "kind": kind,
                "foreign_body_ids": body_ids,
                "sample_denial_lines": sample_lines,
                "full_supersession_triad": full_triad,
                "archived_adjacent": archived_adjacent,
                "exclusions": exclusions,
                "unresolved": not exclusions,
            }
        )

    return {
        "schema": OBSERVATION_SCHEMA,
        "identity_conflicts": identity_conflicts,
        "visa_damage": visa_damage,
        "visa_support": visa_support,
    }


def observation_for_state(state: dict) -> dict:
    """Return the runtime observation, rebuilding old raw-cache states."""
    observation = state.get("terminal_page_observation")
    if observation is not None:
        if (
            not isinstance(observation, dict)
            or observation.get("schema") != OBSERVATION_SCHEMA
        ):
            raise ValueError("invalid terminal page observation")
        return observation
    raw_pages = state.get("raw_pages")
    if raw_pages is not None:
        return observe_primary_pages(state["case_id"], raw_pages)
    # Historical non-dump states have no primary lines to inspect.  Absence of
    # evidence cannot activate any guard.
    return observe_primary_pages(state["case_id"], ())


def trusted_rank1_approval(state: dict) -> bool:
    notes = state.get("doc_notes") or {}
    try:
        rank = int(notes.get("finding_rank", 99))
    except (TypeError, ValueError):
        rank = 99
    return (
        notes.get("finding") == "APPROVED"
        and rank <= 1
        and not notes.get("rank1_conflicts")
    )


def _eligible_guards(
    state: dict,
    prediction: dict,
    detail: dict,
) -> list[dict]:
    observation = observation_for_state(state)
    decision = prediction.get("adjudication")
    reasons = list(detail.get("reasons") or ())
    extracted = set(detail.get("extracted_fields") or ())
    guards = []

    reason_set = set(reasons)
    soft_world_check = (
        "embargoed_world" not in reason_set
        or prediction.get("home_world") in SOFT_WORLDS
    )
    if (
        decision == "DENIED"
        and "visa_class" not in extracted
        and bool(reasons)
        and reason_set <= CONDITIONAL_REASONS
        and soft_world_check
        and observation["visa_damage"]
    ):
        guards.append(
            {
                "guard": G1,
                "evidence": {
                    "reasons": reasons,
                    "visa_extracted": False,
                    "selected_visa": prediction.get("visa_class"),
                    "selected_world": prediction.get("home_world"),
                    "soft_world_check": soft_world_check,
                    "same_line_visa_damage":
                        observation["visa_damage"],
                },
            }
        )

    if decision != "APPROVED":
        return guards

    rank1_approval = trusted_rank1_approval(state)
    unresolved = [
        page
        for page in observation["identity_conflicts"]
        if page.get("unresolved")
    ]
    if unresolved and not rank1_approval:
        guards.append(
            {
                "guard": G2,
                "evidence": {
                    "unresolved_pages": unresolved,
                    "trusted_rank1_approval": False,
                },
            }
        )

    visa = str(prediction.get("visa_class") or "")
    support_pages = observation["visa_support"].get(visa, [])
    if (
        "visa_class" in extracted
        and visa in VISAS
        and visa not in ADVERSE_VISAS
        and bool(support_pages)
        and all(
            page.get("full_supersession_triad")
            for page in support_pages
        )
        and not rank1_approval
    ):
        guards.append(
            {
                "guard": G3,
                "evidence": {
                    "visa": visa,
                    "visa_extracted": True,
                    "support_pages": support_pages,
                    "triad_only": True,
                    "trusted_rank1_approval": False,
                },
            }
        )
    return guards


def apply_terminal_guards(
    state: dict,
    prediction: dict,
    detail: dict,
) -> tuple[dict, dict]:
    """Apply the frozen damage-G1 + G2 + G3 terminal bundle."""
    guards = _eligible_guards(state, prediction, detail)
    if not guards:
        return prediction, detail
    prior = prediction.get("adjudication")
    if prior not in {"APPROVED", "DENIED"}:
        return prediction, detail

    prediction = dict(prediction)
    detail = dict(detail)
    primary = guards[0]["guard"]
    reason = f"terminal_guard_{primary}"
    lock = {
        "schema": LOCK_SCHEMA,
        "prior_adjudication": prior,
        "reason": reason,
        "guards": guards,
    }
    prediction["adjudication"] = "NEEDS_REVIEW"
    detail["reasons"] = [reason]
    detail["path"] = (
        f"NEEDS_REVIEW:{reason}:"
        f"{min(len(detail.get('extracted_fields') or ()), 9)}"
    )
    detail["terminal_guard"] = lock
    if prior == "APPROVED":
        detail["terminal_approval_guard"] = lock
    return prediction, detail
