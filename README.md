# MIB Visible-Evidence OCR Solution

Tyler Gibbs's offline, CPU-only solution for the resubmission to the 8090 MIB document challenge. It reads each PDF packet and writes one adjudication row per document, with no LLM, cloud service, API key, or network access at runtime.

## Quick start

The validation PDFs live outside Git. Follow [data/README.md](data/README.md) to download and unzip them at the repository root, then build and run:

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

The image accepts exactly two arguments: the input PDF directory and output JSONL path. It writes one schema-valid row per PDF and runs with no network.

## How it works

The runtime is a deterministic, one-pass visible-evidence pipeline:

- **Span forensics**: PyMuPDF removes hidden, off-crop, and low-contrast text before OCR or adjudication.
- **Local OCR**: RapidOCR and a compact PP-OCRv5 recognizer recover visible text without an LLM or VLM.
- **Field resolution**: closed-vocabulary parsers and template-anchored ROI readers resolve fields.
- **Adjudication**: a deterministic policy engine, terminal safety guards, and a small exported review resolver produce the decision.
- **Calibrated confidence**: confidence is calibrated from public training data.

The submitted implementation is intentionally the audited one-pass system. Research-only multi-pipeline ensembles were not submitted because they could not meet the official six-seconds-per-PDF average runtime contract. See [SOLUTION.md](SOLUTION.md) for the selection process and local scores.

## Challenge documents

- [SOLUTION.md](SOLUTION.md): how this candidate was selected and evaluated
- [PRD.md](PRD.md): the challenge's business problem and requirements
- [FIELD_MANUAL.md](FIELD_MANUAL.md): adjudication policy excerpt
- [EVALUATION.md](EVALUATION.md): submission files and scoring
- [DOCKER_SUBMISSION.md](DOCKER_SUBMISSION.md): the offline Docker contract

## Development

The public contract tests run in CI ([public-contract.yml](.github/workflows/public-contract.yml)):

```bash
python -m pip install jsonschema
python -m unittest discover -s tests -v
```

## Provenance

The runtime is reused under MIT from the public Calling Moonshots solution at commit `9ed5ed360ae40053dfee80bff09eef29a83a3980`. That project is itself derived under MIT from Brian Pridgen's `handemanai` baseline at commit `4b37a7815bea79de0a01beca6eb6566e1611af73`.

No other participant's validation predictions were copied or used. The resubmission predictions were generated independently by executing this code over the checksum-verified public validation PDFs.

See [ATTRIBUTION.md](ATTRIBUTION.md), [NOTICE.md](NOTICE.md), and [SOLUTION.md](SOLUTION.md) for details.

## License

MIT. See [LICENSE](LICENSE) and [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES/).
