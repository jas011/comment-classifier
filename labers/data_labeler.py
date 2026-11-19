import json
import os
from typing import Literal
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

with open("merged_file.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if len(line.strip()) >= 3]
ds = pd.DataFrame(lines, columns=["text"])


class CommentCharacterization(BaseModel):
    real_query: float = Field(ge=0.0, le=1.0, description="Genuine question, how-to ask, spec/info request, or topic/feature request")
    appreciation: float = Field(ge=0.0, le=1.0, description="Praise, gratitude, positive feedback, or endorsement")
    bad_experience: float = Field(ge=0.0, le=1.0, description="Complaint, defect, bug, error, frustration, or poor experience")
    noise: float = Field(ge=0.0, le=1.0, description="Spam, promo, link drop, bot text, or empty low-signal chatter")
    primary_category: Literal["real_query", "appreciation", "bad_experience", "noise"]
    rationale: str = Field(max_length=160)


SYSTEM_PROMPT = """You characterize user feedback into soft scores [0.0 to 1.0] across four categories:
- real_query: genuine questions, help/clarification requests, spec/pricing/setup asks, or follow-up/feature ideas.
- appreciation: praise, thanks, satisfaction, or positive validation.
- bad_experience: complaints, defects, errors, delays, bad quality, confusion/frustration, or negative outcomes.
- noise: spam, self-promo, irrelevant links, or meaningless filler.

Note that a single text can have multiple non-zero scores (e.g., appreciation + real_query, or appreciation + bad_experience).
Return strictly valid JSON matching the requested schema."""


def label_text(text: str, domain: str = "creator", model: str = "gpt-4o-mini") -> dict:
    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Domain: {domain}\nText: {text}",
            },
        ],
        text_format=CommentCharacterization,
    )
    parsed: CommentCharacterization = response.output_parsed
    return parsed.model_dump()


def extract_comment_text(row: dict) -> str:
    for key in ("text",):
        if key in row and isinstance(row[key], str) and row[key].strip():
            return row[key].strip()
    return ""


def run_youtube_sample(
    num_samples: int = 800,
    output_path: str = "youtube_silver_labels.jsonl",
    model: str = "gpt-4o-mini",
) -> None:
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, row in enumerate(ds.to_dict(orient="records")):
            if idx % 20 != 0:
                continue

            text = extract_comment_text(row)
            if not text:
                continue

            labels = label_text(text, domain="creator", model=model)
            record = {
                "source": "merged_file.txt",
                "domain": "creator",
                "row_index": idx,
                "text": text,
                **labels,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            count += 1
            print(f"[{count}/{num_samples}] {record['primary_category']} -> {text[:60]}")
            if count >= num_samples:
                break


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Please set OPENAI_API_KEY before running.")
    run_youtube_sample(num_samples=850, output_path="youtube_silver_labels.jsonl")
