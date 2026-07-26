# MIB Visible-Evidence Pipeline

This repository contains a reproducible, offline CPU pipeline for the 8090 MIB
document challenge. It combines visible-raster OCR, evidence-aware field
resolution, deterministic safety rules, and small task-specific classifiers.
No LLM, VLM, cloud OCR, API key, or runtime network access is used.

## Architecture

1. **Visible evidence boundary.** PyMuPDF native spans are accepted only when
   they intersect the crop box and have sufficient contrast. White/off-page
   prompt injections are discarded.
2. **Raster OCR.** Every page is rendered at 220 DPI and read by Tesseract.
   Low-quality pages get contrast enhancement and form-line removal. A Poppler
   render is used as an independent fallback because the two rasterizers have
   complementary failure modes on damaged scans.
3. **Evidence resolution.** Closed-vocabulary fields use fuzzy OCR matching;
   names, dates, and sponsor IDs use context-specific parsers. Candidate values
   follow the manual's document precedence, including explicit correction
   notes. Cross-document conflicts become risk evidence. Outcome-determinative
   risk flags are never imputed when their visible evidence is absent.
4. **Hybrid prediction.** A character n-gram text model handles damaged words
   and layout variants, while an Extra Trees model handles the extracted
   structured record. A held-out blend is wrapped in deterministic guardrails
   for manual findings, disqualifying flags, revoked sponsors, transit cases,
   and unpaid fees. Packet identifiers are removed before vectorization.
5. **Calibrated uncertainty.** A second-stage logistic model predicts whether
   the chosen adjudication is correct using out-of-fold model margin,
   disagreement, OCR quality, missing fields, and guardrail activation.

## Research grounding

The bounded multi-render OCR design is a resource-constrained application of
the diversity and voting results reported by Reul et al. in
[Improving OCR Accuracy on Early Printed Books](https://arxiv.org/abs/1802.10038)
and Wick et al. in
[Calamari](https://arxiv.org/abs/1807.02004). Instead of loading a committee of
neural OCR models, the pipeline creates diverse observations with two
rasterizers and targeted image preprocessing, then selects the observation
with the strongest document-specific anchors.

Its selective retry policy also follows the quality-aware routing pattern in
[Benchmarking OCR Pipelines with Adaptive Enhancement](https://arxiv.org/abs/2604.25176):
easy pages stay on the fast path, while low-confidence pages receive bounded
enhancement and independent-render retries. This implementation deliberately
uses classical image operations instead of the paper's CNN so it remains
offline, small, and CPU-efficient.

The confidence layer follows the post-hoc calibration discipline described by
Guo et al. in
[On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599).
It also adopts the explicit prediction-correctness target studied by Liu et al.
in
[Optimizing Calibration by Gaining Aware of Prediction Correctness](https://arxiv.org/abs/2404.13016).
Its inputs and target are generated strictly out of fold, so the submitted
confidence estimates model correctness rather than in-sample class
probability.

## Reproduce training

The public dataset must be unpacked under `data/`.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m solution.train --workers 4
```

The OCR cache in `training_cache/` is resumable and intentionally ignored by
Git. The fitted artifact is `models/model.joblib`.

## Run locally

```bash
.venv/bin/python -m solution.predict data/train /tmp/mib-predictions.jsonl
python3 scripts/validate_submission.py \
  --submission /tmp/mib-predictions.jsonl \
  --manifest data/train_labels.csv \
  --require-complete
```

## Exact offline contract

```bash
docker build -t mib-submission .
docker run --rm --network none \
  --mount type=bind,src="$PWD/data/train",dst=/input,readonly \
  --mount type=bind,src=/tmp/mib-output,dst=/output \
  mib-submission /input /output/predictions.jsonl
```

The entrypoint writes incrementally, uses at most four worker processes, writes
temporary raster fallbacks only below `/tmp`, and requires a read-only
container root.
