#!/usr/bin/env bash
set -euo pipefail

input_dir="${1:?usage: run.sh <input_pdf_dir> <output_path>}"
output_path="${2:?usage: run.sh <input_pdf_dir> <output_path>}"
shift 2

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MIB_NATIVE_SCAN_OCR="${MIB_NATIVE_SCAN_OCR:-1}"
# The submitted profile is score-optimal under the challenge's published
# utility matrix. Keep the library-level resolver fail-closed by default, while
# making this entrypoint's candidate-trained, visible-feature head explicit.
export MIB_REVIEW_MODEL="${MIB_REVIEW_MODEL:-1}"

exec python3 /app/scripts/predict.py "$input_dir" "$output_path" "$@"
