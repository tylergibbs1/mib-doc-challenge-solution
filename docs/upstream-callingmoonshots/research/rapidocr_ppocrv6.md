# RapidOCR and PP-OCRv6 findings

Accessed 2026-07-26 from current tagged primary sources.

Primary sources:

- [RapidOCR v3.9.2](https://github.com/RapidAI/RapidOCR/releases/tag/v3.9.2),
  commit `095232a4c94f7f0e6600ba5bba1177010ad696d4`
- [RapidOCR v3.9.2 model manifest](https://github.com/RapidAI/RapidOCR/blob/v3.9.2/python/rapidocr/default_models.yaml)
- [RapidOCR wheel asset preparation](https://github.com/RapidAI/RapidOCR/blob/v3.9.2/python/tools/prepare_wheel_assets.py)
- [PaddleOCR v3.7.0](https://github.com/PaddlePaddle/PaddleOCR/releases/tag/v3.7.0)
- [PP-OCRv6 introduction](https://www.paddleocr.ai/v3.7.0/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html)
- [PaddleOCR detector models](https://www.paddleocr.ai/v3.7.0/en/version3.x/module_usage/text_detection.html)
- [PaddleOCR recogniser models](https://www.paddleocr.ai/v3.7.0/en/version3.x/module_usage/text_recognition.html)

## Version pins

- Use `rapidocr==3.9.2` for the v6 experiment. v3.9.0 introduced PP-OCRv6
  and v3.9.2 is the current tagged package.
- Pin ONNX Runtime independently. RapidOCR intentionally does not install or
  pin an inference engine.
- Record PaddleOCR v3.7.0 as the upstream model-family release.

PaddleOCR's v3.7 pipeline defaults to medium PP-OCRv6 models. RapidOCR v3.9.2
defaults to small PP-OCRv6 models. “The PP-OCRv6 default” is therefore
ambiguous and must not appear in experiment receipts.

## Candidate artefacts

Values below are the converted ONNX artefacts from RapidOCR's tagged manifest,
not Paddle native model archives.

| Artefact | Bytes | SHA-256 |
| --- | ---: | --- |
| `PP-OCRv6_det_tiny.onnx` | 1,829,618 | `f42c0fbd294d95eac1a550e131b277dac97462c8025fa4b6c3cec1b7894bd3d5` |
| `PP-OCRv6_det_small.onnx` | 9,929,594 | `090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f` |
| `PP-OCRv6_det_medium.onnx` | 62,119,454 | `92078b7355007ccfffcd4c8cd441a3afd4538904d06881b29a155e1e679907c2` |
| `PP-OCRv6_rec_tiny.onnx` | 4,489,813 | `e16e242de5937ad92609223f19bc2aff3727ee40b095f996907c24749bad251b` |
| `PP-OCRv6_rec_small.onnx` | 21,234,383 | `6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884` |
| `PP-OCRv6_rec_medium.onnx` | 76,629,984 | `eef444829dbbe18d7fea59a3f6eb75647518d2b3a9568d27c92e42940204894b` |
| `en_PP-OCRv5_rec_mobile.onnx` | 7,872,351 | `c3461add59bb4323ecba96a492ab75e06dda42467c9e3d0c18db5d1d21924be8` |

The first candidate remains the v6-small detector plus the English PP-OCRv5
mobile recogniser: 17,801,945 bytes combined. This is a hypothesis, not an
upstream performance guarantee. Paddle's English-v5 and multilingual-v6
recogniser metrics come from different benchmarks and are not directly
comparable.

## Offline packaging contract

- RapidOCR's published wheel bundles the v6-small detector, its orientation
  classifier, and the v6-small recogniser.
- Hybrid, tiny, and medium variants need explicit build-time acquisition or a
  custom wheel.
- Use per-stage `model_path` for ONNX files. `model_dir` is for Paddle-style
  model directories.
- A configured `model_path` is checked for existence, not SHA-256. Verify the
  committed provenance manifest during image build and at startup.
- Registry resolution can download missing or corrupt models. Production must
  use explicit local paths so `--network none` never activates a downloader.
- ONNX recognisers embed their character dictionary; no separate keys file is
  required for these official conversions.
- The high-level `RapidOCR` constructor builds classifier state even when
  `Global.use_cls=False`. Retain the classifier artefact unless the high-level
  orchestrator is intentionally bypassed.
- Do not call `result.vis()` unless a local font is packaged and configured;
  its default visualization path may lazily download a font.

## Configuration traps to freeze

- Python `params` requires RapidOCR enum values for engine, language, model
  type, OCR version, and task type. Raw strings are for YAML, not the Python
  constructor.
- `Det.limit_side_len=736` with `limit_type=min` is a shorter-side minimum and
  may upscale an image. It is not a 736-pixel detector cap.
- v3.9.2 defaults `Global.use_preprocess_img=true` and
  `Global.use_vertical_padding=true`.
- ONNX Runtime thread defaults are not the baseline's one-thread values. Set
  intra-op and inter-op threads explicitly to one.
- Freeze preprocessing, vertical padding, max-side length, limit type, and
  limit length identically across model variants.
- RapidOCR has no application-level fallback to legacy OCR. The wrapper must
  implement and test that behavior.

## Naming and provenance

- Paddle names: `PP-OCRv6_small_det` and `PP-OCRv6_small_rec`.
- RapidOCR registry keys:
  `multi_PP-OCRv6_det_small` and `multi_PP-OCRv6_rec_small`.
- RapidOCR filenames:
  `PP-OCRv6_det_small.onnx` and `PP-OCRv6_rec_small.onnx`.
- `v6_full_small` and `v6_det_v5_rec` are local experiment aliases only.

Both RapidOCR v3.9.2 and PaddleOCR v3.7.0 are Apache-2.0. Provenance must
identify Paddle's upstream family, RapidOCR's ONNX conversion and exact file
hash, ONNX Runtime and wheel hash, and both license/notice layers.
