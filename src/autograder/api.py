"""
FastAPI placeholder for AutoGrader.

Full API (upload, evaluate, result, report) will be implemented in Phase 4.
"""

from fastapi import FastAPI

app = FastAPI(
    title="AutoGrader",
    description="Automatic Handwritten Answer Sheet Evaluator — Phase 1 placeholder",
    version="0.1.0",
)


@app.get("/health")
def health():
    """Health check for deployment."""
    return {"status": "ok", "phase": 1}
