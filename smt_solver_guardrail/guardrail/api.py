"""FastAPI app exposing the guardrail endpoint."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .extract import Extractor
from .models import GuardrailReport, GuardrailRequest
from .pipeline import TypeCheckFailure, run_guardrail

app = FastAPI(title="SMT Solver Guardrail", version="0.1.0")
_extractor = Extractor()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/guardrail", response_model=GuardrailReport)
def guardrail(req: GuardrailRequest) -> GuardrailReport:
    try:
        return run_guardrail(req, _extractor)
    except TypeCheckFailure as e:
        raise HTTPException(status_code=422, detail={"type_errors": e.errors}) from e
    except Exception as e:  # malformed compile target, LLM failure, etc.
        raise HTTPException(status_code=422, detail=f"guardrail failed: {e}") from e
