import os
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from inference.inference import FeedbackService

MODEL_DIR = os.getenv("COMMENT_MODEL_DIR", "./distilbert_comment_model")

app = FastAPI(
    title="Neuro-Fuzzy Comment & Feedback API",
    version="1.0.0",
    description="Scores text with DistilBERT and routes feedback via Mamdani fuzzy logic.",
)

service = FeedbackService(model_dir=MODEL_DIR)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    domain: Literal["creator", "product", "service"] = "creator"
    impact: float = Field(default=0.5, ge=0.0, le=1.0)
    item_id: Optional[str] = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_dir": MODEL_DIR}


@app.post("/analyze")
def analyze_comment(req: AnalyzeRequest) -> dict:
    return service.process_item(
        text=req.text,
        domain=req.domain,
        impact=req.impact,
        item_id=req.item_id,
    )


@app.get("/queue")
def get_queue(exclude_noise: bool = Query(default=True)) -> dict:
    items = service.get_queue(exclude_noise=exclude_noise)
    return {"count": len(items), "items": items}


@app.post("/resolve/{item_id}")
def resolve_item(item_id: str) -> dict:
    updated = service.resolve_item(item_id)
    if not updated:
        raise HTTPException(status_code=404, detail="Item not found")
    return updated