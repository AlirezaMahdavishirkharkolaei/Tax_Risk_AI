"""REST API for self-hosted deployments.

Run with ``taxriskscan serve`` (or ``uvicorn taxriskscan.api:app``). Interactive
documentation is served at ``/docs``.

Environment variables
---------------------
TAXRISKSCAN_API_KEYS       comma-separated keys; when set, every request except
                           /health must send one in the ``X-API-Key`` header
TAXRISKSCAN_CONFIG         path to a TOML configuration
TAXRISKSCAN_MODEL          path to a model saved with ``taxriskscan train --save``
TAXRISKSCAN_TRAINING       'real' (default), 'synthetic' or a labelled CSV, if no model file
TAXRISKSCAN_MAX_UPLOAD_MB  upload limit (default 20)

Uploaded files are processed in memory and not stored.
"""

from __future__ import annotations

import hmac
import io
import os
from contextlib import asynccontextmanager
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from . import __version__, model
from .config import DEFAULT_CONFIG, ConfigError, load_config
from .metrics import DataError
from .pipeline import analyze

STATE: dict = {}


def _settings():
    cfg_path = os.environ.get("TAXRISKSCAN_CONFIG")
    config = load_config(cfg_path) if cfg_path else DEFAULT_CONFIG
    model_path = os.environ.get("TAXRISKSCAN_MODEL")
    if model_path:
        models = model.load(model_path)
    else:
        models = model.train(os.environ.get("TAXRISKSCAN_TRAINING", "real"), config)
    keys = [k.strip() for k in os.environ.get("TAXRISKSCAN_API_KEYS", "").split(",") if k.strip()]
    limit = int(float(os.environ.get("TAXRISKSCAN_MAX_UPLOAD_MB", "20")) * 1024 * 1024)
    return config, models, keys, limit


@asynccontextmanager
async def lifespan(app: FastAPI):
    config, models, keys, limit = _settings()
    STATE.update(config=config, models=models, keys=keys, limit=limit)
    yield
    STATE.clear()


app = FastAPI(
    title="TaxRiskScan API",
    version=__version__,
    description="Tax-avoidance, profit-shifting and Pillar Two screening for Country-by-Country Reporting data. "
                "Results are risk indicators, not legal or tax conclusions.",
    lifespan=lifespan,
)


def require_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    keys = STATE.get("keys") or []
    if not keys:
        return
    if not x_api_key or not any(hmac.compare_digest(x_api_key, k) for k in keys):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")


Auth = Depends(require_key)


async def _read_upload(file: UploadFile) -> io.BytesIO:
    raw = await file.read(STATE["limit"] + 1)
    if len(raw) > STATE["limit"]:
        raise HTTPException(status_code=413, detail="File too large.")
    if not raw:
        raise HTTPException(status_code=422, detail="The uploaded file is empty.")
    buf = io.BytesIO(raw)
    name = (file.filename or "").lower()
    looks_xml = name.endswith(".xml") or raw.lstrip()[:5] in (b"<?xml", b"<cbc:") or raw.lstrip()[:1] == b"<"
    buf.name = "upload.xml" if looks_xml else "upload.csv"
    return buf


def _run(data, fiscal_year: int):
    try:
        return analyze(data, config=STATE["config"], models=STATE["models"], fiscal_year=fiscal_year)
    except (DataError, ConfigError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


class Entity(BaseModel):
    jurisdiction: str
    revenue: float
    profit_before_tax: float
    tax_paid: float | None = None
    tax_accrued: float | None = None
    employees: float
    group: str | None = None
    year: int | None = None
    tangible_assets: float | None = None
    payroll_costs: float | None = None


class AnalyzeRequest(BaseModel):
    entities: list[Entity] = Field(..., min_length=1, max_length=50_000)
    fiscal_year: int = 2026


FiscalYear = Annotated[int, Query(ge=2015, le=2040, description="Fiscal year for Pillar Two if the data has no 'year'")]


@app.get("/health", tags=["service"])
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/model", tags=["service"], dependencies=[Auth])
def model_card() -> dict:
    """What the model was trained on, how well it performs, and the data terms."""
    return STATE["models"].card()


@app.get("/config", tags=["service"], dependencies=[Auth])
def config() -> dict:
    c = STATE["config"]
    return {**c.to_dict(), "fingerprint": c.fingerprint()}


@app.post("/analyze", tags=["analysis"], dependencies=[Auth])
async def analyze_file(file: UploadFile = File(..., description="CSV or OECD CbC XML"), fiscal_year: FiscalYear = 2026) -> dict:
    """Score an uploaded CSV or OECD CbC XML file."""
    return _run(await _read_upload(file), fiscal_year).to_dict()


@app.post("/analyze/json", tags=["analysis"], dependencies=[Auth])
def analyze_json(req: AnalyzeRequest) -> dict:
    """Score entities sent as JSON (for system-to-system integration)."""
    df = pd.DataFrame([e.model_dump() for e in req.entities])
    df = df.dropna(axis=1, how="all")
    return _run(df, req.fiscal_year).to_dict()


@app.post("/report", tags=["analysis"], response_class=HTMLResponse, dependencies=[Auth])
async def report(file: UploadFile = File(...), fiscal_year: FiscalYear = 2026,
                 title: str = Query("TaxRiskScan report", max_length=120)) -> str:
    """Full self-contained HTML audit report for an uploaded file."""
    from .report import html_report
    return html_report(_run(await _read_upload(file), fiscal_year), title=title)


@app.post("/pillar2", tags=["analysis"], dependencies=[Auth])
async def pillar_two(file: UploadFile = File(...), fiscal_year: FiscalYear = 2026) -> dict:
    """Pillar Two screening only (Transitional CbCR Safe Harbour and top-up estimate)."""
    return _run(await _read_upload(file), fiscal_year).to_dict()["pillar2"]


@app.get("/template.csv", tags=["service"], response_class=PlainTextResponse)
def template() -> str:
    from .cli import TEMPLATE
    return TEMPLATE
