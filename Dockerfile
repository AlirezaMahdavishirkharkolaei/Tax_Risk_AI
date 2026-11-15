# TaxRiskScan REST API - self-hosted, no data leaves your infrastructure.
#   docker build -t taxriskscan .
#   docker run -p 8000:8000 -e TAXRISKSCAN_API_KEYS=change-me taxriskscan
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY taxriskscan ./taxriskscan
RUN pip install ".[api]" && useradd --create-home --uid 10001 app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1
CMD ["uvicorn", "taxriskscan.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
