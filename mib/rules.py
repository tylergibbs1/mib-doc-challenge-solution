"""MIB adjudication policy, mined from 1,000 labeled training cases.

Explains 97.3% of training adjudications from true field values with zero
APPROVED/DENIED confusions; the residual is document-evidence-driven
NEEDS_REVIEW handled by the evidence-quality gate, not by these rules.

Sources: FIELD_MANUAL.md plus rules inferred from labeled examples (documented
in the technical memo): extra revoked sponsors, the Wolf-1061c soft embargo,
unpaid-fee denial applying even to DIP-1, and DIP-1's exemptions.
"""
from datetime import date

from .vocab import DISQUALIFYING_FLAGS, REVIEW_FLAGS

REVOKED_SPONSORS = frozenset({
    # Listed in the public field manual:
    "SPN-0007", "SPN-0139", "SPN-4040",
    # Inferred from labeled examples (11-15 consistent denials each):
    "SPN-7331", "SPN-2718", "SPN-9090",
})
SOFT_EMBARGO_WORLDS = frozenset({"Wolf-1061c"})
# Hard-embargo worlds deny regardless of visa class: 50/50 training cases
# DENIED including all 13 DIP-1 packets (every one carries planetary_embargo,
# but the flag text is frequently unreadable while the world name is legible
# trusted evidence — the world itself is the documented proxy).
HARD_EMBARGO_WORLDS = frozenset({"Eris Relay", "TRAPPIST-1e"})
STALE_DAYS = 180


def adjudicate(fields, receipt_date=None):
    """Apply the mined policy to extracted fields.

    Returns (decision, reasons). Deny rules run first; DIP-1 is exempt from
    sponsor, embargo-world, and staleness rules (but not fees or risk flags).
    """
    flags = set(fields.get("risk_flags", "none").split("|")) - {"none", ""}
    visa = fields.get("visa_class") or ""
    is_dip = visa == "DIP-1"
    deny, review = [], []

    if flags & DISQUALIFYING_FLAGS:
        deny.append("disqualifying_flag")
    if fields.get("fee_status") == "unpaid":
        deny.append("unpaid_fee")
    if visa == "TRANSIT-7":
        deny.append("transit_visa")
    if fields.get("home_world") in HARD_EMBARGO_WORLDS:
        deny.append("embargoed_world")
    if fields.get("home_world") in SOFT_EMBARGO_WORLDS and not is_dip:
        deny.append("embargoed_world")
    if fields.get("sponsor_id") in REVOKED_SPONSORS and not is_dip:
        deny.append("revoked_sponsor")
    if receipt_date and fields.get("arrival_date") and not is_dip:
        try:
            arrival = date.fromisoformat(fields["arrival_date"])
            if (receipt_date - arrival).days > STALE_DAYS:
                deny.append("stale_application")
        except ValueError:
            review.append("unparseable_date")

    if fields.get("fee_status") == "unknown":
        review.append("fee_unknown")
    if flags & REVIEW_FLAGS:
        review.append("review_flag")
    if not fields.get("visa_class"):
        review.append("missing_visa")
    if not fields.get("sponsor_id") and not is_dip:
        review.append("missing_sponsor")

    if deny:
        return "DENIED", deny
    if review:
        return "NEEDS_REVIEW", review
    return "APPROVED", ["clean"]


def optimal_decision(p_approved, p_denied, p_review):
    """Expected-value-optimal decision under the challenge's scoring matrix.

    Raw points: correct 8, review-hedge 2, missed-review 1, wrong A/D 0,
    false approval -4. Never approve on a coin flip; DENIED never goes negative.
    """
    ev = {
        "APPROVED": 8 * p_approved + p_review - 4 * p_denied,
        "DENIED": 8 * p_denied + p_review,
        "NEEDS_REVIEW": 2 + 6 * p_review,
    }
    return max(ev, key=ev.get), ev
