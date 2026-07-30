# Official challenge contract

Accessed 2026-07-26 from the official challenge repository at commit
`38ce8883dea9f87c27a8a95f134e54fe8b673064`.

Primary sources:

- <https://github.com/8090-inc/mib-doc-challenge/blob/38ce8883dea9f87c27a8a95f134e54fe8b673064/README.md>
- <https://github.com/8090-inc/mib-doc-challenge/blob/38ce8883dea9f87c27a8a95f134e54fe8b673064/EVALUATION.md>
- <https://github.com/8090-inc/mib-doc-challenge/blob/38ce8883dea9f87c27a8a95f134e54fe8b673064/FIELD_MANUAL.md>
- <https://github.com/8090-inc/mib-doc-challenge/blob/38ce8883dea9f87c27a8a95f134e54fe8b673064/DOCKER_SUBMISSION.md>
- <https://github.com/8090-inc/mib-doc-challenge/blob/38ce8883dea9f87c27a8a95f134e54fe8b673064/scripts/run_docker_submission.py>
- <https://huggingface.co/datasets/arjun-krishna1/mib-doc-challenge-data>

## Facts that constrain the design

- Scoring is 80 classification, 50 extraction, and 20 calibration points.
- A false approval of a truly denied case receives `-4` raw classification
  points. A correct decision receives `8`; review hedges receive partial credit.
- Confidence is scored against whether the emitted adjudication is exactly
  correct. It is not the probability of approval.
- Runtime is offline, CPU-only, 4 vCPU, 8 GiB RAM, read-only root filesystem,
  writable `/tmp`, and six seconds per PDF on average.
- The 5,000-case validation run has a 30,000-second hard stop. The uncompressed
  image cap is 4 GiB; each model artefact is capped at 250 MiB and all model
  artefacts together at 1 GiB.
- Hidden white text, off-crop text, fake answer keys, and barcode instructions
  are untrusted. Visible evidence precedence starts with a visible adjudicator
  stamp or signed note.
- Private validation labels include admin-only unrecoverable-field metadata.
  The public train labels intentionally do not.
- Final ranking includes a fully private test and manual code review for
  hardcoded answers, filename dependence, per-case editing, reproducibility,
  and generalisation to new layouts.

## Data identity

- Archive: `mib-doc-challenge-public-data-v2026-07-07.zip`
- Repository-declared SHA-256:
  `a9bb8c1bbf51346ebf49c2e3e1acdb7a5d6cd0760162767b0d133c7b7200f3c4`
- Server-reported size on 2026-07-26: `2,879,970,504` bytes.
- Contents: 1,000 labelled training PDFs and 5,000 unlabelled validation PDFs.

## Reproduction implications

- Use the official runner rather than reproducing its Docker flags manually.
- Pass the already built pinned image with `--image-tag ... --skip-build`; this
  prevents a later dirty working tree from silently changing the baseline.
- Use distinct output paths for deterministic reruns. The official runner
  unlinks an existing output before starting.
- Evaluate only after format validation, and retain both `evaluation.json` and
  `case_scores.jsonl`.
- A score-only gate is insufficient. Every experiment must also report
  catastrophic false approvals, runtime, output completeness, and changed
  decisions.
