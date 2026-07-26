FROM python:3.12-slim-trixie

ENV DEBIAN_FRONTEND=noninteractive \
    OMP_THREAD_LIMIT=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends poppler-utils tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY solution /app/solution
COPY models /app/models
COPY run.sh /app/run.sh
RUN chmod 0555 /app/run.sh

ENTRYPOINT ["/app/run.sh"]
