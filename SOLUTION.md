# Reproducible solution

## Selection discipline

The previous submitted system was frozen before this resubmission. Candidate
changes were evaluated with a public-training-only 700/150/150 author split,
group-disjoint out-of-fold folds, and two author-held-out partitions.

A four-system research ensemble scored about 140.9 locally but was rejected:
running all of its OCR systems could not satisfy the official average runtime
limit. Two attempted one-pass distillations were also rejected because both
lost score on untouched author holdouts. This repository therefore publishes
the strongest reproducible one-pass candidate rather than cached or
unreproducible ensemble outputs.

On the two untouched 150-case author partitions, the imported candidate scored
131.19 and 137.43. Pooled, it scored about 134.31 compared with about 125.56
for the frozen previous submission, a local improvement of roughly 8.75
points. The paired document bootstrap for that comparison had a positive lower
bound. These are local public-training estimates, not private leaderboard
claims.

## Runtime architecture

1. Inspect PDF spans for visibility before rasterization. Hidden and off-crop
   instructions are never treated as evidence.
2. Run bounded low-resolution OCR, with higher-resolution recovery only when
   deny-relevant evidence is missing.
3. Parse closed vocabularies and structured identifiers, preserving
   cross-page authority and cancellation evidence.
4. Apply deterministic field-manual policy and one-way terminal guards.
5. Resolve only a narrowly defined insufficient-evidence review subset with a
   frozen, dependency-free tree ensemble.
6. Emit calibrated confidence and an atomic output file.

The image contains no LLM or VLM and requires no runtime network. Model
artifacts are below the per-model and aggregate limits. The upstream retained
image measured 4.761 seconds per PDF over all 5,000 validation documents with
four CPUs, and this resubmission keeps that runtime closure unchanged.

## Compliance boundaries

- Training labels influence only public-training-derived parsers, resolver,
  and calibration artifacts.
- Case identifiers and open applicant/sponsor identities are excluded from the
  learned review resolver.
- The filename supplies the output case ID; no prediction lookup table exists.
- The validation PDFs are unlabeled and are used only as runtime input.
- No validation prediction file from another participant is present in the
  Docker context or copied into the submission.
- `.dockerignore` allowlists only the runtime source and model artifacts.

The original upstream technical memo and model-provenance notes are retained
under `docs/upstream-callingmoonshots/` for reviewer audit.
