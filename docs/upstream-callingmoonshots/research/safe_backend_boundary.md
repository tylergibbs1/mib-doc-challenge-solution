# M0 safe OCR boundary

## Security claim

The new type is an accidental-misuse boundary inside the application. It is not
a Python sandbox. A malicious future module could still import filesystem or
PDF libraries or read the worker's `MIB_ACTIVE_CASE` environment value, so code
review and dependency tests remain part of the gate. Strong isolation would
require a scrubbed subprocess and pixel-only IPC; that is outside M0.

“Sanitised” means authorised by the current forensics policy. It does not mean
every image is hidden-bbox-masked: the current native path may use a proven
embedded scan, while an ineligible native candidate falls back to a conforming
composited render.

The existing `ImageViewRegistry` remains post-inference, fail-soft provenance.
It does not issue, store, or authorise backend inputs.

## M0 compatibility rule

Current production calls to `mib.ocr.ocr_page` remain untouched in M0. The new
modules are unreachable from default execution, so baseline call count, array
identity, retry order, observer behavior, exceptions, and serialization cannot
drift before a measured experiment opts in.

## Candidate contract

- There is no unauthenticated public “mark these pixels safe” function. The
  trusted forensics/pipeline bridge obtains a private issuer callable for one
  issuance attempt; that callable is consumed even when the attempt is
  invalid. Python privacy is a code-review convention, not proof of
  sanitisation or isolation.
- A candidate backend accepts only `SanitizedImageView`.
- The view contains the exact two-dimensional `uint8` pixels and narrow source
  metadata. It contains no path, case ID, PDF document, or page object.
- The candidate receives a bytes-backed, isolated, read-only C-order snapshot
  so mutation cannot affect legacy fallback. The caller's array remains
  writable.
- Issued inputs are single-use. The dispatcher rejects caller mutation between
  issue and dispatch, and rechecks before fallback; callers must retain
  sequential ownership while one candidate attempt is running.
- Backend output is validated as OCR `(text, confidence)` pairs.
- Empty OCR is a valid result, not a fallback trigger.
- Candidate selection, initialization, execution, or result-contract
  exceptions fall back exactly once to the legacy adapter.
- Catch `Exception`, not `BaseException`; process cancellation, timeout, and
  termination signals must retain their existing semantics.
- If legacy OCR fails, propagate its exception unchanged.
- The legacy adapter delegates to the existing
  `ocr.ocr_page(original_array, min_lines=..., hq=...)` using the untouched
  original object.
- The PP-OCRv6 placeholder has no import-time model initialization, download,
  or production registration. Explicit selection while unavailable raises a
  bounded backend-unavailable exception and falls back to legacy.
- Output identity cannot prove that an experimental candidate ran: a failed
  candidate may safely fall back to legacy. M1 bakeoff receipts must separately
  record candidate attempts, successes, and fallback reasons before candidate
  scores are interpreted.

## Tests required before integration

- Raw arrays, paths, PDF documents, and page objects are rejected by the
  candidate dispatcher.
- Candidate pixels are read-only and detached from the legacy input.
- Direct construction, source-label laundering, handle reuse, and
  issue-to-dispatch mutation are rejected.
- Valid empty output does not fall back.
- Bad selection, initialization, execution, and malformed output each invoke
  legacy exactly once.
- `BaseException` escapes without invoking legacy.
- A legacy exception remains the observed exception.
- Importing the v6 placeholder performs no model load or download.
- Default production continues to call the unchanged legacy entrypoint.
- Raw prediction SHA-256 and `cmp` equality remain the integration gate;
  canonical JSON equality alone is insufficient.

## Runtime identity closure

The upstream identity verifier omitted nine files copied by the Dockerfile:

- `mib/caseid.py`
- `mib/feeread.py`
- `mib/flagread.py`
- `mib/native_ledger.py`
- `mib/noteread.py`
- `mib/sponsorread.py`
- `mib/two_ledger.py`
- `mib/worldread.py`
- `models/reason_buckets.json`

M0 now includes those nine files in the immutable upstream closure and checks
the Docker COPY sources against the explicit current closure. After the three
new OCR modules are added, only the current closure expands; the pinned
upstream tuple remains immutable so that its producer stays independently
checkable.

The frozen upstream build also generated 17
`/app/mib/__pycache__/*.cpython-311.pyc` files while importing the pipeline.
They are executable image state, not Docker-context source files. Runtime
manifest v2 therefore binds them separately by exact path, source-module
association, size, and image hash. The post-M0 Dockerfile disables bytecode
writes during its build so new candidate images can prove a source-only
`/app` closure.

The final post-M0 census proved exactly 36 source-backed files, zero
image-only files, and zero symlinks. The complete dependency-backed test run
reported 1,205 passed and 51 declared skips. The real PDF/scan hidden-twin
boundary subset passed 68/68 in the pinned offline runtime.
