# Handeman baseline findings

Accessed 2026-07-26 from the public solution repository at commit
`4b37a7815bea79de0a01beca6eb6566e1611af73`.

Primary sources:

- <https://github.com/handemanai/mib-doc-challenge-solution/blob/4b37a7815bea79de0a01beca6eb6566e1611af73/README.md>
- <https://github.com/handemanai/mib-doc-challenge-solution/blob/4b37a7815bea79de0a01beca6eb6566e1611af73/MEMO.md>
- <https://github.com/handemanai/mib-doc-challenge-solution/blob/4b37a7815bea79de0a01beca6eb6566e1611af73/mib/forensics.py>
- <https://github.com/handemanai/mib-doc-challenge-solution/blob/4b37a7815bea79de0a01beca6eb6566e1611af73/mib/ocr.py>
- <https://github.com/handemanai/mib-doc-challenge-solution/blob/4b37a7815bea79de0a01beca6eb6566e1611af73/mib/view_registry.py>
- <https://github.com/handemanai/mib-doc-challenge-solution/blob/4b37a7815bea79de0a01beca6eb6566e1611af73/NOTICE.md>

## What must remain invariant

- PDF spans are classified before rasterisation. The historical composited
  path masks hidden regions before enhancement can resurrect them.
- A direct native-scan path is authorised only by a fail-closed PDF-object and
  paint-order audit. If authorisation fails, the code uses a conforming
  composited render.
- Native and composited evidence remain separate until explicit two-ledger
  reconciliation. Native-path abstention never weakens the baseline ledger.
- Hidden content is distrust metadata only and can move a result away from
  approval, never toward it.
- OCR is adaptive: low-resolution first, then selective HQ and rotation
  retries. ONNX/OpenCV/BLAS threading is pinned to one thread per worker.
- Policy is deterministic and confidence calibration is downstream of the
  decision. Calibration cannot alter fields or adjudication.

## Verified upstream claims to reproduce, not assume

- Fixed MD5 split: 799 development cases and 201 sealed holdout cases.
- Reported scores: 129.52 development and 126.46 holdout.
- Reported catastrophic false approvals: zero on holdout.
- Reported scoring-contract runtime: 3.41 seconds/PDF.
- Reported image and model footprint: 0.27 GiB image and 12 MB model artefacts.

These are upstream claims until this checkout produces its own receipts.

## Current OCR boundary

`mib/ocr.py` constructs RapidOCR internally and accepts a two-dimensional
`uint8` array. The full-page detector is the bundled PP-OCRv4 model; the
recogniser is overridden with the committed English PP-OCRv5 mobile ONNX
artefact. Fast and HQ engines differ only in resolution limits.

`mib/view_registry.py` is currently a fail-soft diagnostic sink. It fingerprints
pixels after they have already been selected or consumed. It does not prevent a
future backend from reopening a PDF. M0 therefore needs a separate typed
safe-view input contract; the diagnostic registry should remain decision
neutral.

## Narrow M0 insertion point

- Preserve `ocr.ocr_page(...)` as the public legacy path.
- Put the existing engine construction and retry ladder behind a legacy backend
  adapter with no changed defaults.
- Make new backends accept only a safe image-view object created by the
  forensics/pipeline boundary. Do not give a backend a PDF path, document, or
  page object.
- Keep a PP-OCRv6 module importable but unreachable from production defaults.
- On any backend-selection or backend-execution error, explicitly return to the
  legacy path.
- Prove the refactor with byte-identical predictions, existing red-team tests,
  and tests that the dispatcher never supplies a backend with an unsanitised
  view. This is not a sandbox: an in-process Python module could still
  construct its own data source or access the filesystem, so code review and
  dependency controls remain part of the boundary.

## Corrections to the initial gameplan

- The upstream split is not exactly 800/200; the deterministic MD5 rule
  materialises 799/201.
- The existing image-view registry is evidence provenance, not an access-control
  registry. Treating it as a safe input registry without code changes would be
  a false security claim.
- PyMuPDF is AGPL-3.0 or commercially licensed. The upstream notice states that
  distributing the built image engages AGPL obligations. Provenance work must
  preserve this fact.
