# Technical Memo — Visible-Evidence Hybrid

## Approach

My submission treats the rendered page as the trust boundary. PDF-native text
is useful, but this dataset demonstrates why it cannot be trusted blindly:
white text, off-crop content, and hidden answer keys can all be present in the
text layer without appearing to a reviewer. I inspect native spans and retain
only spans that intersect the crop box and have sufficient contrast. Every page
is also rendered at 220 DPI and OCRed with Tesseract. The OCR text is therefore
derived from visible pixels and cannot see white-on-white prompt injections.

The OCR stage has bounded, selective fallbacks. Ordinary pages use a grayscale
PyMuPDF render. Low-confidence pages are retried with contrast enhancement and
form-line removal. PyMuPDF and Poppler rasterize some damaged synthetic scans
differently, so hard pages receive one independent Poppler render and the
higher-quality OCR result wins. The system uses four processes and one
Tesseract thread per process. It writes predictions incrementally and runs
under the supplied read-only, network-disabled Docker contract.

Field extraction is evidence-aware rather than a single regex over concatenated
text. Pages are classified as intake forms, biometric slips, sponsor letters,
registry extracts, fee receipts, or manual notes. Candidate values receive the
document precedence specified in the field manual. Explicit manual corrections
outrank the original field. Closed vocabularies (species, home world, visa,
purpose, fee, and risk flags) use OCR-tolerant matching; names, dates, and
sponsor IDs use context-specific parsers and conservative OCR-character
repairs. Cross-document disagreements are retained as evidence instead of being
silently overwritten.

The adjudicator is a hybrid of deterministic policy and two small offline
models. A character n-gram logistic model is robust to recurring OCR damage
such as `DENIEN`, while an Extra Trees model consumes the resolved structured
record. Small field classifiers fill non-risk closed-vocabulary fields only
when the rule extractor is missing or the model is highly confident. Risk flags
must come from visible evidence and are never inferred from packet
correlations. Case identifiers are also removed before vectorization. Hard
guardrails cover visible manual findings, disqualifying risks, revoked
sponsors, transit classes, stale non-diplomatic packets, and unpaid fees.
Manual findings are applied first, which prevents an old or crossed-out denial
from overriding a later signed decision.

I selected the text/structured blend and decision thresholds using stratified
five-fold out-of-fold predictions. Model selection maximized the challenge's
actual asymmetric classification score subject to no more than five
catastrophic false approvals across the 1,000 public cases. Confidence is not
the winning class probability. A separate logistic calibrator is trained on
out-of-fold correctness using model margin, model disagreement, OCR quality,
missing fields, and guardrail activation.

## Validation discipline and results

The validation PDFs and their case IDs were never used as labels, pseudo-labels,
or model-selection feedback. The pipeline contains no per-case lookup table and
does not use filenames for anything beyond the required case ID.

On five-fold out-of-fold public training predictions, the selected policy
achieved 74.3% adjudication accuracy, 63.26/80 classification points,
14.99/20 calibration points, and five catastrophic false approvals. Closed
field extraction is strongest on species (93.3%), home world (89.5%), purpose
(88.9%), visa class (86.4%), and arrival date (83.6%). The fitted full-data
smoke evaluation scores 134.94/150 with zero false approvals; I report it only
as an integration check, not as a generalization estimate.

Tests cover the public evaluator/schema contract, white and off-page text
rejection, multi-page field resolution, manual-note precedence, and stale
packet policy. The fitted artifact is about 17 MB and the Docker image is about
255 MB, both comfortably inside the published limits. A complete 1,000-packet
exact-contract Docker run took 23 minutes 38 seconds (1.42 seconds per PDF),
and the final anti-overfit rebuild passed a read-only, network-disabled parity
smoke without nested parallelism or runtime warnings.

## Known failure modes

- Severe scan damage can destroy an arbitrary sponsor ID, date, or applicant
  name. The system uses a schema-valid unknown fallback and lowers confidence,
  but exact extraction credit is still lost.
- Rare risk classes (`memory_tampering` and `active_warrant`) have few public
  examples. Rules protect high-precision visible hits; learned field models
  never invent a risk flag when its evidence is absent.
- Multi-applicant packets remain difficult when the active form's case header
  is itself destroyed. Document precedence helps, but I do not claim perfect
  entity resolution.
- OCR behavior changes slightly between macOS Tesseract 5.5 and Debian
  Tesseract 5.3. The final submission is tested inside the exact Docker image,
  and character n-grams reduce sensitivity to those changes.
- The public manual is intentionally incomplete. I inferred only stable,
  repeated sponsor and stale-packet behavior; ambiguous cases are routed to
  review rather than accumulating case-specific exceptions.

## With another week

I would train inside the final Linux image from the start, add coordinate-aware
OCR for the fixed form fields, and use nested cross-validation for field-model
thresholds. I would also add a small image-quality model for illegible
biometrics, because raster damage is sometimes more informative than OCR text.
Finally, I would profile fallbacks by damage type and spend the recovered
runtime on a second OCR pass only where held-out expected value is positive.
