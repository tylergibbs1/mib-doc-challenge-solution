FROM python:3.11-slim

# System libs for OpenCV (pulled in by RapidOCR) on a slim base.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*

# Offline runtime: all dependencies and OCR models are baked into the image.
# No LLM/VLM, no torch/paddle; nothing here follows instructions, so the
# injection surface the dataset targets does not exist in this system.
# Every resolved Python distribution is pinned to the version in the retained,
# score-validated image. In particular, RapidOCR and ONNX Runtime otherwise
# leave most of their dependency graph floating; a clean rebuild could silently
# change image preprocessing, model execution, or serialization behavior.
RUN pip install --no-cache-dir \
    coloredlogs==15.0.1 \
    flatbuffers==25.12.19 \
    humanfriendly==10.0 \
    mpmath==1.3.0 \
    pymupdf==1.28.0 \
    rapidocr-onnxruntime==1.4.4 \
    onnxruntime==1.20.1 \
    rapidfuzz==3.14.5 \
    numpy==2.2.6 \
    opencv-python==4.11.0.86 \
    packaging==26.2 \
    pillow==12.3.0 \
    protobuf==7.35.1 \
    pyclipper==1.4.0 \
    PyYAML==6.0.3 \
    shapely==2.1.2 \
    six==1.17.0 \
    sympy==1.14.0 \
    tqdm==4.69.1

WORKDIR /app
# Keep the copied application closure source-only. The upstream baseline image
# generated /app bytecode during its build; M0 records that historical state,
# while new candidate images avoid untracked executable copies.
ENV PYTHONDONTWRITEBYTECODE=1
COPY mib/ /app/mib/
COPY models/ /app/models/
COPY scripts/predict.py scripts/run_shard.py /app/scripts/
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

# Trigger RapidOCR model unpack at build time so runtime needs no writes
# outside /tmp, then verify the pipeline imports cleanly.
RUN python -c "from rapidocr_onnxruntime import RapidOCR; RapidOCR()" && \
    python -c "import sys; sys.path.insert(0, '/app'); import mib.pipeline"

ENV TMPDIR=/tmp
ENTRYPOINT ["/app/run.sh"]
