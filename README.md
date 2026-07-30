# Tyler Gibbs — MIB Visible-Evidence OCR Solution

This repository contains the offline CPU solution used for Tyler Gibbs's
resubmission to the 8090 MIB document challenge.

The runtime is a deterministic visible-evidence pipeline:

- PyMuPDF span forensics remove hidden, off-crop, and low-contrast text before
  OCR or adjudication.
- RapidOCR and a compact PP-OCRv5 recognizer recover visible text without an
  LLM, VLM, cloud service, API key, or runtime network access.
- Closed-vocabulary parsers and template-anchored ROI readers resolve fields.
- A deterministic policy engine, terminal safety guards, and a small exported
  review resolver produce the adjudication.
- Confidence is calibrated from public training data.

The submitted implementation is intentionally the audited one-pass system.
Research-only multi-pipeline ensembles were not submitted because they could
not meet the official six-seconds-per-PDF average runtime contract.

## Build and run

```bash
docker build -t mib-tylergibbs .
mkdir -p /tmp/mib-output
docker run --rm --network none \
  --cpus 4 \
  --memory 8g \
  --read-only \
  --tmpfs /tmp:rw,size=2g \
  --mount type=bind,src="$PWD/data/validation",dst=/input,readonly \
  --mount type=bind,src=/tmp/mib-output,dst=/output \
  mib-tylergibbs /input /output/predictions.jsonl
```

The image accepts exactly two arguments: the input PDF directory and output
JSONL path. It writes one schema-valid row per PDF and runs with no network.

## Provenance

The runtime is reused under MIT from the public Calling Moonshots solution at
commit `9ed5ed360ae40053dfee80bff09eef29a83a3980`. That project is itself derived
under MIT from Brian Pridgen's `handemanai` baseline at commit
`4b37a7815bea79de0a01beca6eb6566e1611af73`.

No other participant's validation predictions were copied or used. The
resubmission predictions are generated locally by executing this code over the
checksum-verified public validation PDFs.

See [ATTRIBUTION.md](ATTRIBUTION.md), [NOTICE.md](NOTICE.md), and
[SOLUTION.md](SOLUTION.md) for details.
