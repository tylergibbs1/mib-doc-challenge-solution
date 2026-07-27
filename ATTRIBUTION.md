# Attribution

This solution is original work built on the public challenge materials. One
implementation idea was adapted after reviewing public MIT-licensed challenge
solutions:

- The separation between policy-time missing values and output-only
  categorical priors was inspired by the `OUTPUT_ONLY_FALLBACKS` boundary in
  [Abhishek21g/mib-doc-challenge-solution](https://github.com/Abhishek21g/mib-doc-challenge-solution),
  which attributes its render-first stack and fallback approach to Chris
  Strobl's public MIT solution.
- The targeted orientation retry for scans whose raster rotation disagrees
  with PDF metadata was prompted by the public MIT-licensed
  [naidx0 challenge solution](https://github.com/naidx0/mib-doc-challenge).
  The implementation and quality gate here are independent and use this
  repository's OCR confidence and visible form-anchor scoring.
- Recovering signed findings from notes whose damaged headings defeat the
  page-type classifier was prompted by the public MIT-licensed
  [rupaut98 challenge solution](https://github.com/rupaut98/mib-doc-solution).
  This implementation independently uses a two-stage heading and finding
  match and leaves the underlying page type unchanged.
- The narrow DIP-1 positive-evidence certificate was prompted by the public
  MIT-licensed
  [Abhishek21g challenge solution](https://github.com/Abhishek21g/mib-doc-challenge-solution/tree/08947bd72b0058d5803d9bfe08428cd7311485b7).
  This implementation independently requires visible `$809` fee evidence,
  unique intake/registry name agreement, and no conflicting guardrail; it
  never reads or copies another submission's predictions.

This repository reimplements that boundary independently. Its priors are
learned from the public training split, generated out of fold for local
evaluation, and never influence adjudication or confidence. No validation
predictions or case-specific answers from another participant are used.
