# Model provenance

This file records runtime model artefacts and planned experiment artefacts.
Hashes are SHA-256. An artefact marked `planned` is not shipped or reachable
from the default runtime.

## Baseline image

- Producer source:
  `4b37a7815bea79de0a01beca6eb6566e1611af73`
- Image:
  `sha256:bf10110b0fec73b33e9cf2c0ed06801346cd7c93a2b34f69cf170f6156e6c4be`
- Architecture: `linux/arm64`
- Built 2026-07-26
- Full image-inspect receipt:
  `evaluation/baseline-image-inspect.json` (SHA-256
  `6c173bdacda2c54bfbe1234a8f63229765596a10e49793aae72a2a21d6630b80`)
- Frozen baseline receipt:
  `evaluation/baseline_manifest.json` (SHA-256
  `45949f6b48b6680a1b4a9bc853914d3d1d11bf5f7bde4ed5452be88bed3a8c8c`)

The image ID is the reproducibility boundary for M0. A future rebuild from the
same Git commit is not promised to be byte-identical: the Dockerfile uses a
floating base tag, apt packages are not version-pinned, and transitive Python
packages are not wheel-hash locked. The saved `pip freeze` inventory records
what was installed in this image; it does not make a rebuild hermetic.

The exact `/app` census is frozen in
`evaluation/baseline-runtime-manifest.json` (SHA-256
`9e8d67809318f32af6a383749c205dd8cc15daf1325bccab60fd30990456f429`).
It binds all 33 source-backed Docker `COPY` files and all 17 additional
CPython bytecode files created when the Dockerfile imported `mib.pipeline`
during the build. Those bytecode files are recorded separately as
build-generated executable image state; they are not silently ignored or
misrepresented as tracked source. `tools/capture_runtime_manifest.py`
reproduces the offline, read-only, no-mount exact-path and exact-hash census.

## M0 scaffold image

- Producer source:
  `68518291a3b9523d5a2c2064b5940517f2dbb1f8`
- Image:
  `sha256:2b844d9017286d984e0ef9385f4f6f261be8f0f442df163ace883654cd49aa8f`
- Architecture: `linux/arm64`
- Size: 286,782,044 bytes
- Entrypoint: `/app/run.sh`

The live, offline, read-only, no-mount census bound all 36 Docker-copied files
byte-for-byte to the producer commit. It found zero image-only files, zero
symlinks, and no unlisted path beneath `/app`. The full evidence is embedded
in `evaluation/m0-post-scaffold-identity/summary.json`.

The official 1,000-case runner used the immutable image ID, `--network none`,
four CPUs, 8 GiB RAM, a read-only root, and the two-argument entrypoint. Its
predictions, aggregate evaluation, and case scores were raw byte-identical to
the accepted upstream baseline:

| Artefact | Rows | SHA-256 |
| --- | ---: | --- |
| `predictions.jsonl` | 1,000 | `9d0deadec06671596b490d60ff3b5a6e396c34867903776017f8f9dd4a9847de` |
| `evaluation.json` | - | `9c9085336aee3dc1e8dfef727ccf51af41a3605e464946b154bcdbf7c16bbf08` |
| `case_scores.jsonl` | 1,000 | `1348ed97b98d3fc88f7a3a9463ec3f44d341f7486036a15658f66b25c951abc3` |

The official score remained `128.82639727555556`, with one catastrophic false
approval. The M0 modules remain unreachable from default production; this
control proves the scaffold did not alter the existing OCR path.

## Final candidate image

- Image: `sha256:420a3f51a1471e3a6b6e901b2924faf1ce499355311e589e55c3ef8dec8bc8ec`
- Size: 287,080,971 bytes
- Architecture: `linux/arm64`
- Offline full-1,000 score: `134.28832794222222`
- Original-image full-1,000 score: `128.82639727555556`
- Catastrophic false approvals: 13 candidate / 1 original
- Sealed-150 score: `136.40276041481482`
- Repeated sealed prediction SHA-256:
  `caf8475d60eca82ba2dee8c455695f81f2284b7b644a9799d4b20285b6703cbb`

The full and sealed runs used the official runner with no network, four CPUs,
8 GiB RAM and a read-only root. The shallow review resolver is the principal
classification gain and the principal safety trade-off; the official totals
above already include every false-approval penalty.

### Third-party OCR artefacts

| Exact image path | Bytes | SHA-256 | Source | Licence | Runtime role |
| --- | ---: | --- | --- | --- | --- |
| `/app/models/en_PP-OCRv5_rec_mobile.onnx` | 7,872,351 | `c3461add59bb4323ecba96a492ab75e06dda42467c9e3d0c18db5d1d21924be8` | PaddleOCR English PP-OCRv5 mobile, RapidOCR ONNX conversion | Apache-2.0 | Active recogniser |
| `/usr/local/lib/python3.11/site-packages/rapidocr_onnxruntime/models/ch_PP-OCRv4_det_infer.onnx` | 4,745,517 | `d2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9` | Bundled by `rapidocr-onnxruntime==1.4.4` | Apache-2.0 | Active detector |
| `/usr/local/lib/python3.11/site-packages/rapidocr_onnxruntime/models/ch_PP-OCRv4_rec_infer.onnx` | 10,857,958 | `48fc40f24f6d2a207a2b1091d3437eb3cc3eb6b676dc3ef9c37384005483683b` | Bundled by `rapidocr-onnxruntime==1.4.4` | Apache-2.0 | Present but overridden |
| `/usr/local/lib/python3.11/site-packages/rapidocr_onnxruntime/models/ch_ppocr_mobile_v2.0_cls_infer.onnx` | 585,532 | `e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c` | Bundled by `rapidocr-onnxruntime==1.4.4` | Apache-2.0 | Present; execution disabled |

The bundled artefact hashes were measured inside the pinned image under
`--network none`. They are not committed files. ONNX Runtime reports the active
English recogniser's embedded `character` metadata as 1,416 UTF-8 bytes with
SHA-256
`e025a66d31f327ba0c232e03f407ae8d105e1e709e7ccb3f408aa778c24e70d6`.
That embedded metadata is the exact recognition dictionary; there is no
separate active keys file. The overridden bundled recogniser's embedded
dictionary is 26,249 UTF-8 bytes with SHA-256
`28b2362ad4ab2dc38769aa72feb535e3a9ddb3fd2a7585a05920e6393b1dc7f7`.

### Candidate-produced artefacts

| Artefact | Bytes | SHA-256 | Runtime role |
| --- | ---: | --- | --- |
| `models/calibrator.json` | 11,528 | `438d0116d16ca5e997cc72db65e63026f7f6b38ef3e13184fc1df465a244f973` | Active |
| `models/confusion_costs.json` | 1,594 | `4e5818f9a36cb30152c81ee28c97ae3f789e732a016cdc2b0a7aa07590d41e58` | Active |
| `models/name_vocab.json` | 3,045 | `928270fb38026ae640d541df9575ca26abccd3c641fdf2f71c7e00c3ef792718` | Active |
| `models/path_confidence.json` | 3,524 | `246413a3b5f0de6e92635bb7de164767f23abf00d7a5a0092ca2b252da979e5d` | Active |
| `models/pix_bank.npz` | 1,704,139 | `1fdabeacca8c6e4133957bc834c729041a51b8b2b9644952c861ce27adc57823` | Active |
| `models/reason_buckets.json` | 3,323 | `834af4da2b37d36c4ef49d1ed7c2b79d56de2c72b0decb1b02aa888e9d38e68b` | Active |
| `models/review_resolver.json` | 1,583,393 | `3a0677b0fa22fc50ee0e297fe23b23f6a81c54393ebe2da354df27c389cc106f` | Active; pure-Python shallow-forest resolver for `insufficient_evidence` reviews |
| `models/transducer_enc.int8.onnx` | 1,206,266 | `09c5a4a1d8f92f90ab2c086a743043bcafc5fb25ecb72696ee343756ca17bff6` | Shipped disabled |
| `models/transducer_dec.int8.onnx` | 1,755,035 | `b1539a1cd82e20c5784d4696a61836e71475c7d83c4adeef8e19b3b22f531fb6` | Shipped disabled |
| `models/transducer_vocab.json` | 624 | `3484ad770148357cd92f98e6f6f4fbb6ab2da2201ab45dcc52adebda16ce178b` | Shipped disabled |

Derivation and licensing details for these baseline artefacts remain in
[`NOTICE.md`](../NOTICE.md).

## Planned PP-OCRv6 experiment artefacts

Source manifest:
<https://github.com/RapidAI/RapidOCR/blob/v3.9.2/python/rapidocr/default_models.yaml>.
RapidOCR v3.9.2 and PaddleOCR v3.7.0 are Apache-2.0.

| Artefact | Bytes | SHA-256 | Status |
| --- | ---: | --- | --- |
| `PP-OCRv6_det_tiny.onnx` | 1,829,618 | `f42c0fbd294d95eac1a550e131b277dac97462c8025fa4b6c3cec1b7894bd3d5` | planned |
| `PP-OCRv6_det_small.onnx` | 9,929,594 | `090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f` | planned |
| `PP-OCRv6_det_medium.onnx` | 62,119,454 | `92078b7355007ccfffcd4c8cd441a3afd4538904d06881b29a155e1e679907c2` | planned |
| `PP-OCRv6_rec_tiny.onnx` | 4,489,813 | `e16e242de5937ad92609223f19bc2aff3727ee40b095f996907c24749bad251b` | planned |
| `PP-OCRv6_rec_small.onnx` | 21,234,383 | `6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884` | planned |
| `PP-OCRv6_rec_medium.onnx` | 76,629,984 | `eef444829dbbe18d7fea59a3f6eb75647518d2b3a9568d27c92e42940204894b` | planned |

No planned artefact may be promoted merely because its file exists. Promotion
requires verified hash, explicit local path, offline smoke test, grouped-CV and
sealed-holdout gates, zero new catastrophic false approvals, and runtime
compliance.
