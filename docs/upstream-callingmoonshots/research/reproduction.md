# M0 reproduction procedure

Pinned source identities:

- Official challenge:
  `38ce8883dea9f87c27a8a95f134e54fe8b673064`
- Handeman baseline:
  `4b37a7815bea79de0a01beca6eb6566e1611af73`
- Dataset SHA-256:
  `a9bb8c1bbf51346ebf49c2e3e1acdb7a5d6cd0760162767b0d133c7b7200f3c4`

The sibling checkout layout is intentional. Development tooling finds the
official checkout at `../mib-doc-challenge`, while the baseline Docker context
excludes data.

## Full 1,000-case scoring-contract run

Build the clean source once, then make the official runner reuse that exact
image:

```bash
docker build \
  --label org.opencontainers.image.revision=4b37a7815bea79de0a01beca6eb6566e1611af73 \
  -t mib-intake:4b37a7815bea79de0a01beca6eb6566e1611af73 .

python3 ../mib-doc-challenge/scripts/run_docker_submission.py \
  --repo . \
  --input-dir ../mib-doc-challenge/data/train \
  --output evaluation/upstream-train-run-2-solo/predictions.jsonl \
  --manifest ../mib-doc-challenge/data/train_labels.csv \
  --timeout-seconds 6000 \
  --image-tag sha256:bf10110b0fec73b33e9cf2c0ed06801346cd7c93a2b34f69cf170f6156e6c4be \
  --skip-build \
  --require-complete \
  --cpus 4 \
  --memory 8g
```

The independent accepted twin uses
`evaluation/upstream-train-run-4-solo/` and the same immutable image ID. The
official runner deletes an existing output path before starting, so every
accepted run starts with a nonexistent prediction path. Run 3 was interrupted
for a user-requested host restart at 410 durable cases and is explicitly
excluded; it has no success receipt.

Run the two full reproductions sequentially on this Docker Desktop host.
Launching two 4-vCPU containers concurrently reduced each run below the pace
needed for its independent 6,000-second timeout. The incomplete concurrent
attempt was interrupted. The completed run that overlapped it is also
excluded: raw comparison showed generic fallback defaults for
`MIB-000222`, `MIB-000261`, and `MIB-000989`, while the clean solo execution
completed those cases normally.

Both qualifying solo runs produced prediction SHA-256
`9d0deadec06671596b490d60ff3b5a6e396c34867903776017f8f9dd4a9847de`
and official score `128.82639727555556`. Their predictions, official
evaluation, and case-score payloads are raw byte-identical. Each accepted
receipt binds a distinct container ID, a disjoint execution interval, the
official runner transcript, Docker completion event, and clean hashes for the
runner, validator, and evaluator. The frozen
`evaluation/baseline_manifest.json` has SHA-256
`45949f6b48b6680a1b4a9bc853914d3d1d11bf5f7bde4ed5452be88bed3a8c8c`.

## Post-scaffold identity control

The committed M0 source
`68518291a3b9523d5a2c2064b5940517f2dbb1f8` produced immutable image
`sha256:2b844d9017286d984e0ef9385f4f6f261be8f0f442df163ace883654cd49aa8f`.
Before scoring, the strict live census proved 36/36 exact source-backed files,
zero image-only files, and zero symlinks beneath `/app`.

The full acceptance command was:

```bash
python3 tools/run_ocr_bakeoff.py \
  --variant legacy_control \
  --repo . \
  --challenge-dir ../mib-doc-challenge \
  --input-dir ../mib-doc-challenge/data/train \
  --truth ../mib-doc-challenge/data/train_labels.csv \
  --corpus-input-dir ../mib-doc-challenge/data/train \
  --corpus-truth ../mib-doc-challenge/data/train_labels.csv \
  --split-manifest splits/v1.json \
  --split all \
  --output-dir evaluation/m0-post-scaffold-identity \
  --image-ref mib-intake:m0-68518291a3b9523d5a2c2064b5940517f2dbb1f8 \
  --expected-image-id \
    sha256:2b844d9017286d984e0ef9385f4f6f261be8f0f442df163ace883654cd49aa8f \
  --expected-source-sha 68518291a3b9523d5a2c2064b5940517f2dbb1f8 \
  --m0-identity-check \
  --baseline-predictions \
    evaluation/upstream-train-run-2-solo/predictions.jsonl \
  --timeout-seconds 6000 \
  --cpus 4 \
  --memory 8g
```

The official runner completed in 3,652.254540 seconds; total bakeoff time was
3,653.041529 seconds. It validated 1,000 records with zero missing, extra,
duplicate, or invalid rows. Raw `cmp` equality and all three frozen output
hashes passed. The score remained `128.82639727555556`, including one
catastrophic false approval. The complete receipt is
`evaluation/m0-post-scaffold-identity/summary.json` (SHA-256
`fe3d2bf293663b836c78f999a2547dfd8bbd1df8f1b23023ae9e531bd4cbfb81`).

The final repository-wide suite ran inside the dependency-complete image under
`--network none` with read-only source mounts and test-only pytest/Git fixture
mounts: 1,205 passed and 51 declared skips.

## Freeze-C scope

The grouped manifest uses only the 700 `train_fit` cases. Calibration and
sealed holdout are not opened, hashed into the grouped inventory, or assigned
to folds. It is a partial grouped proxy: 261/700 cases currently belong to
non-singleton layout groups and 439 remain singleton assignments. This clears
the frozen 25% non-singleton-coverage floor but is not evidence that most cases
have a known layout twin. Two independent offline generations in the pinned
baseline image were raw byte-identical; `splits/group_folds_v1.json` has
SHA-256
`cf117637faad20c2cd3bd5a367d5bc8b2c81fbf7699e9270ed1cc552f5b54211`.

PyMuPDF 1.28.0 reproduced the repository's known cumulative `none_dealloc`
failure when all PDFs were opened sequentially in one interpreter. Freeze-C
therefore uses spawned workers recycled every 25 PDFs, one PDF per
multiprocessing chunk, and a bounded worker timeout. Each PDF's size and
SHA-256 must also match before and after feature extraction.

Raw text and pixels are not retained. Visible heading content and coarse
sanitized-render statistics can still influence group membership.

## Published 799/201 split

The upstream score claim uses:

```python
int(md5(case_id).hexdigest(), 16) % 5 == 0
```

as holdout. On the public labels this yields 799 development cases and 201
holdout cases. It is distinct from a full-training score.

The current `scripts/eval_split.py` is host-oriented and needs the same pinned
Python packages as the Dockerfile. Do not silently compare its split scores
with a full 1,000-case Docker score.

## Known documentation defects

- The pasted gameplan built a labelled image and then called the official
  runner without `--skip-build`; that would build and run a second image.
- The pasted command omitted `--require-complete`.
- This machine exposes `python3`, not `python`.
- `SUBMISSION.md` references validator and data paths as though they were local
  to the solution repository. In this checkout they live in the sibling
  challenge repository.
- The README's identity example declares the 799-case `dev-md5` partition but
  mounts all 1,000 train PDFs. Current preflight hashes the mounted directory,
  so that example cannot produce a matching bound identity without first
  materialising the declared partition.

These defects affect reproduction instructions, not the runtime prediction
path. M0 receipts use the corrected commands above.

The saved package inventory is `pip freeze --all` output, not a wheel-hash
lock, so it identifies the installed environment but does not make a future
rebuild hermetic. The saved image receipt is the complete raw
`docker image inspect` payload for the immutable accepted image.
