# Third-Party Notices and Artifact Provenance

This file records every third-party component this solution redistributes or
links against, and states the origin of each file under `models/`. Tyler Gibbs
reuses the runtime closure from the public Calling Moonshots repository at
commit `9ed5ed360ae40053dfee80bff09eef29a83a3980`; references below to produced
artifacts refer to that upstream project, not to newly claimed authorship.

## Upstream solution baseline

The reused Calling Moonshots runtime is derived from Brian Pridgen's public
[`handemanai/mib-doc-challenge-solution`](https://github.com/handemanai/mib-doc-challenge-solution)
at commit `4b37a7815bea79de0a01beca6eb6566e1611af73`, used under its MIT
license. The original copyright and MIT permission notice are preserved in
`LICENSE`. Calling Moonshots' subsequent source, model, test, and documentation
changes are distributed under the same MIT terms.

That MIT license applies to repository source, not to the third-party
components listed below.

## Redistributed third-party artifact

| File | Component | Upstream | License |
| --- | --- | --- | --- |
| `models/en_PP-OCRv5_rec_mobile.onnx` (7.5 MB) | PP-OCRv5 English mobile text-recognition model | PaddleOCR / PaddlePaddle, exported to ONNX | Apache-2.0 |

This file is redistributed unmodified, and is the only third-party *artifact*
committed to this repository. Its PaddlePaddle origin is self-evident from the
file: the ONNX graph name is `PaddlePaddle Graph in PIR mode`, and it carries
the upstream English character dictionary in its `character` metadata field.

## Runtime dependencies (installed at image build; not vendored here)

Every Python distribution resolved in the retained, score-validated image is
directly pinned in `Dockerfile`. Versions are exact because the score must
reproduce from a clean checkout; installed package metadata and license members
remain included in the image.

| Package | Version | License |
| --- | --- | --- |
| PyMuPDF | 1.28.0 | **AGPL-3.0** (dual-licensed: GNU AGPL 3.0 or Artifex commercial) |
| rapidocr-onnxruntime | 1.4.4 | Apache-2.0 |
| onnxruntime | 1.20.1 | MIT |
| rapidfuzz | 3.14.5 | MIT |
| numpy | 2.2.6 | BSD-3-Clause |
| opencv-python | 4.11.0.86 | Apache-2.0 |
| coloredlogs | 15.0.1 | MIT |
| flatbuffers | 25.12.19 | Apache-2.0 |
| humanfriendly | 10.0 | MIT |
| mpmath | 1.3.0 | BSD-3-Clause |
| packaging | 26.2 | Apache-2.0 OR BSD-2-Clause |
| Pillow | 12.3.0 | MIT-CMU; wheel carries additional native-library notices |
| protobuf | 7.35.1 | BSD-3-Clause |
| pyclipper | 1.4.0 | MIT; embedded Clipper core uses BSL-1.0 |
| PyYAML | 6.0.3 | MIT |
| Shapely | 2.1.2 | BSD-3-Clause; bundled GEOS uses LGPL-2.1 |
| six | 1.17.0 | MIT |
| sympy | 1.14.0 | BSD-3-Clause |
| tqdm | 4.69.1 | MPL-2.0 AND MIT |

`rapidocr-onnxruntime` bundles its own detection/classification ONNX models
(`ch_PP-OCRv4_det`, `ch_PP-OCRv4_rec`, `ch_ppocr_mobile_v2.0_cls`), also
PaddleOCR-derived and Apache-2.0. They are used for text detection; the
recognizer above replaces the bundled `rec` model.

Base image `python:3.11-slim` plus `libgl1`, `libglib2.0-0`, `libgomp1` from
Debian, under their respective upstream licenses.

### On PyMuPDF and AGPL-3.0

PyMuPDF is AGPL-3.0. The **built image** therefore combines MIT-licensed code
with an AGPL-3.0 library, and the image as a distributed whole is governed
by AGPL-3.0 terms. The corresponding source is this repository, which is public,
so the source-availability obligation is satisfied by construction. Our own
files remain MIT (a permissive license is compatible with inclusion in an
AGPL-licensed whole); the combined distribution is what carries AGPL terms.

## Artifacts produced by the upstream authors (`models/`)

None of these came from a third party, and none is keyed to a case ID or
contains an answer. All were derived from the public training data
(`data/train/` + `data/train_labels.csv`) that the challenge provides.

| File | What it is | Derived from |
| --- | --- | --- |
| `calibrator.json` | Logistic + isotonic confidence calibrator coefficients | out-of-fold fit on dev evidence features |
| `path_confidence.json` | Per-decision-path empirical accuracy priors (84 paths) | dev decision paths |
| `reason_buckets.json` | Per-reason-bucket accuracy shrink targets (23 buckets) | dev decision reasons |
| `review_resolver.json` | Five-seed ensemble of 1,500 shallow decision trees for score-optimal resolution of `insufficient_evidence` reviews; contains no case IDs or open applicant/sponsor identity values | 700-case frozen fit split, with grouped out-of-fold and disjoint holdout validation |
| `name_vocab.json` | Applicant-name syllable lexicon (first/last token sets) | train label strings |
| `pix_bank.npz` | Empirical pixel-template bank for the closed-vocabulary decoder | real line crops from the harvest half of dev |
| `confusion_costs.json` | Character-level edit costs (ins/del/sub) | OCR confusion pairs mined from the upstream system's output |
| `transducer_enc.int8.onnx`, `transducer_dec.int8.onnx`, `transducer_vocab.json` | OCR-correction transducer, trained from scratch by us; **ships disabled** (`MIB_TRANSDUCER=0`) | mined + synthetic confusion pairs |

### A note on `pix_bank.npz` keys

The bank is keyed `v|{label}|{value}|{i}` — a crop of the pixels that spell
`{value}` under field label `{label}`. No key contains a case ID, and the
artifact holds templates for exactly the ten field labels the decoder queries.

This is worth stating explicitly because it was not always true. Because
`tools/pixharvest.py` harvests every field label it can verify, earlier revisions
also banked crops for the `Case ID:` field, whose *value* is a case ID string —
228 entries covering 177 training case IDs. They were harvest byproduct and
unreachable at run time (no code path queries the `Case ID:` label, and
`case_id` is taken from the PDF filename stem, never from a pixel match), but a
shipped artifact that greps as a per-PDF table is not worth defending, so they
were removed. Dropping them is output-neutral, verified by byte-identical
extraction states over a 20-case sample including the five slowest packets in
the validation set.

See `MEMO.md` for the decoder's actual role.
