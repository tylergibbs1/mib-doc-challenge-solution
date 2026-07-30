"""Closed vocabularies observed in training data, with fuzzy snapping.

Every extraction target except sponsor_id/arrival_date/case_id comes from a small
legal set; names are 2 tokens from a syllable-combinatoric generator. Snapping OCR
output to the nearest legal value converts OCR noise into a confidence signal:
the snap margin (distance to best vs runner-up) feeds the calibrator.
"""
import json
import re
from pathlib import Path

from rapidfuzz import fuzz, process

SPECIES = [
    "ALPHA_DRACONIAN", "ANDROMEDAN", "AQUARIAN_MANTIS", "ARCTURIAN",
    "CENTAURI_SYNTH", "JOVIAN_GASFORM", "KAIJU_MICRO", "LUNA_SECURID",
    "ORION_GRAYS", "SIRIUS_AVIAN", "TRIANGULAN", "VENUSIAN_MYCELIAL",
]
WORLDS = [
    "Barnard-c", "Eris Relay", "Europa Station", "Gliese-581g", "Kepler-186f",
    "Luyten-b", "Mars Dome-7", "Proxima-b", "Sirius Outpost", "TRAPPIST-1e",
    "Titan Freeport", "Wolf-1061c", "Zeta Reticuli",
]
PURPOSES = [
    "archive audit", "cultural exchange", "diplomatic", "field repair",
    "medical consult", "reactor maintenance", "research", "transit",
    "translation", "xenobotany",
]
VISAS = ["DIP-1", "MED-3", "TRANSIT-7", "XW-1", "XW-2"]
FEES = ["paid", "waived", "unpaid", "unknown"]
FLAGS = [
    "memory_tampering", "planetary_embargo", "active_warrant", "biohazard_red",
    "identity_conflict", "sponsor_mismatch", "illegible_biometrics", "rescinded_denial",
]
DISQUALIFYING_FLAGS = frozenset(FLAGS[:4])
REVIEW_FLAGS = frozenset(FLAGS[4:])

SPONSOR_RE = re.compile(r"SPN-\d{4}")
CASE_RE = re.compile(r"MIB-\d{6}")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


# OCR-confusion edit costs mined from synthetic renders of known vocab values
# passed through the real OCR engine under the corpus damage transforms — no
# labeled dev data involved. Costs are coarse tiers (0.2 common / 0.45 seen /
# 0.7 rare) rather than exact frequencies, so the table can't overfit the
# mining run. Used only to re-rank near-ties between legal values.
_COSTS_PATH = Path(__file__).resolve().parents[1] / "models" / "confusion_costs.json"
_COSTS = json.loads(_COSTS_PATH.read_text()) if _COSTS_PATH.exists() else None


def _weighted_sim(read, legal):
    """Confusion-weighted Levenshtein similarity (0-100): legal -> read using
    the mined OCR channel costs. Cheap substitutions (I->l, rn-shape, case
    slips) barely count; unexplained edits cost full."""
    subs, dels, inss = _COSTS["subs"], _COSTS["dels"], _COSTS["inss"]
    a, b = legal, read
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0.0] * lb
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cb = b[j - 1]
            if ca == cb:
                sc = 0.0
            else:
                sc = subs.get(ca + cb)
                if sc is None:
                    sc = 0.3 if ca.lower() == cb.lower() else 1.0
            cur[j] = min(prev[j - 1] + sc,
                         prev[j] + dels.get(ca, 1.0),
                         cur[j - 1] + inss.get(cb, 1.0))
        prev = cur
    dist = prev[lb]
    denom = max(la, lb) or 1
    return 100.0 * max(0.0, 1.0 - dist / denom)


def snap(value, choices, min_score=70, rerank=True):
    """Snap a noisy read to the nearest legal value.

    Returns (best_value_or_None, score, margin). Margin = best minus runner-up
    score; a low margin means the read was ambiguous between two legal values.
    When the top WRatio candidates are nearly tied, the OCR-confusion-weighted
    distance breaks the tie (an "XW-Z" read is a cheap 2->Z slip from XW-2 but
    an expensive 1->Z slip from XW-1, which uniform costs cannot see).
    risk_flags callers pass rerank=False: re-ranking a garbled token away from
    a disqualifying flag would destroy deny evidence — same directional ban as
    the transducer's trigger-removal rule.
    """
    if not value:
        return None, 0.0, 0.0
    matches = process.extract(value, choices, scorer=fuzz.WRatio, limit=3)
    if not matches or matches[0][1] < min_score:
        return None, matches[0][1] if matches else 0.0, 0.0
    best = matches[0]
    runner = matches[1][1] if len(matches) > 1 else 0.0
    if rerank and _COSTS and len(matches) > 1 and best[1] - matches[1][1] < 12:
        near = [m for m in matches if best[1] - m[1] < 12]
        ranked = sorted(((_weighted_sim(str(value), m[0]), m) for m in near),
                        key=lambda x: -x[0])
        if ranked[0][1][0] != best[0]:
            best = ranked[0][1]
        wmargin = ranked[0][0] - (ranked[1][0] if len(ranked) > 1 else 0.0)
        return best[0], best[1], min(best[1] - runner if best == matches[0]
                                     else wmargin, wmargin)
    return best[0], best[1], best[1] - runner
