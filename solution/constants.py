from __future__ import annotations

OUTPUT_FIELDS = (
    "case_id",
    "applicant_name",
    "species_code",
    "home_world",
    "visa_class",
    "sponsor_id",
    "arrival_date",
    "declared_purpose",
    "risk_flags",
    "fee_status",
    "adjudication",
    "confidence",
)

SPECIES_CODES = (
    "ALPHA_DRACONIAN",
    "ANDROMEDAN",
    "AQUARIAN_MANTIS",
    "ARCTURIAN",
    "CENTAURI_SYNTH",
    "JOVIAN_GASFORM",
    "KAIJU_MICRO",
    "LUNA_SECURID",
    "ORION_GRAYS",
    "SIRIUS_AVIAN",
    "TRIANGULAN",
    "VENUSIAN_MYCELIAL",
)

HOME_WORLDS = (
    "Barnard-c",
    "Eris Relay",
    "Europa Station",
    "Gliese-581g",
    "Kepler-186f",
    "Luyten-b",
    "Mars Dome-7",
    "Proxima-b",
    "Sirius Outpost",
    "Titan Freeport",
    "TRAPPIST-1e",
    "Wolf-1061c",
    "Zeta Reticuli",
)

VISA_CLASSES = ("XW-1", "XW-2", "DIP-1", "MED-3", "TRANSIT-7")

PURPOSES = (
    "archive audit",
    "cultural exchange",
    "diplomatic",
    "field repair",
    "medical consult",
    "reactor maintenance",
    "research",
    "transit",
    "translation",
    "xenobotany",
)

RISK_FLAGS = (
    "active_warrant",
    "biohazard_red",
    "identity_conflict",
    "illegible_biometrics",
    "memory_tampering",
    "planetary_embargo",
    "rescinded_denial",
    "sponsor_mismatch",
)

DISQUALIFYING_FLAGS = frozenset(
    {"active_warrant", "biohazard_red", "memory_tampering", "planetary_embargo"}
)
REVIEW_FLAGS = frozenset(
    {"identity_conflict", "illegible_biometrics", "rescinded_denial", "sponsor_mismatch"}
)

# Three are public policy. The latter three are consistently revoked in every
# labeled example and are deliberately inferable exceptions in the incomplete manual.
REVOKED_SPONSORS = frozenset(
    {"SPN-0007", "SPN-0139", "SPN-4040", "SPN-2718", "SPN-7331", "SPN-9090"}
)

DOCUMENT_PRIORITIES = {
    "manual": 6.0,
    "intake": 5.0,
    "biometric": 4.0,
    "fee": 4.0,
    "sponsor": 3.0,
    "registry": 2.0,
    "unknown": 1.0,
}

ADJUDICATIONS = ("APPROVED", "DENIED", "NEEDS_REVIEW")
