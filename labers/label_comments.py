import argparse
import json
import os
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
from pydantic import BaseModel, Field

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


class CommentCharacterization(BaseModel):
    real_query: float = Field(
        ge=0.0,
        le=1.0,
        description="Genuine question, how-to ask, spec/info request, or topic/feature request",
    )
    appreciation: float = Field(
        ge=0.0,
        le=1.0,
        description="Praise, gratitude, positive feedback, or endorsement",
    )
    bad_experience: float = Field(
        ge=0.0,
        le=1.0,
        description="Complaint, defect, bug, error, frustration, or poor experience",
    )
    noise: float = Field(
        ge=0.0,
        le=1.0,
        description="Spam, promo, link drop, bot text, or empty low-signal chatter",
    )
    primary_category: Literal["real_query", "appreciation", "bad_experience", "noise"]
    rationale: str = Field(max_length=160)


SYSTEM_PROMPT = """You characterize user feedback into soft scores [0.0 to 1.0] across four categories:
- real_query: genuine questions, help/clarification requests, spec/pricing/setup asks, or follow-up/feature ideas.
- appreciation: praise, thanks, satisfaction, or positive validation.
- bad_experience: complaints, defects, errors, delays, bad quality, confusion/frustration, or negative outcomes.
- noise: spam, self-promo, irrelevant links, or meaningless filler.

Note that a single text can have multiple non-zero scores (e.g., appreciation + real_query, or appreciation + bad_experience).
Return strictly valid JSON matching the requested schema."""


def build_combined_series(
    df: pd.DataFrame,
    cols: list[str],
    sep: str,
    include_headers: bool = False,
) -> pd.Series:
    """Combine selected DataFrame columns row-wise, skipping empty values."""
    combined = pd.Series("", index=df.index, dtype="object")

    for col in cols:
        part = df[col].fillna("").astype(str).str.strip()
        if include_headers:
            part = part.where(part.str.len() == 0, f"{col}: " + part)

        has_prior = combined.str.len() > 0
        has_part = part.str.len() > 0
        combined = (
            combined.where(~(has_prior & has_part), combined + sep + part)
            .where(~(~has_prior & has_part), part)
        )

    return combined.str.strip()


def load_input_dataframe(
    input_path: str,
    columns: list[str] | None = None,
    sep: str = "\n",
    include_headers: bool = False,
    min_len: int = 3,
) -> tuple[pd.DataFrame, list[str], str]:
    """Load a .csv, .tsv, or .txt file and build a combined '_combined_text' column."""
    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(input_path)
    elif ext == ".tsv":
        df = pd.read_csv(input_path, sep="\t", on_bad_lines="skip", low_memory=False)
    elif ext == ".txt":
        with open(input_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if len(line.strip()) >= min_len]
        cols = columns if columns else ["text"]
        if len(cols) > 1:
            raise ValueError("TXT input only supports a single column name.")
        df = pd.DataFrame({cols[0]: lines})
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Expected '.csv', '.tsv', or '.txt'.")

    if ext in (".csv", ".tsv"):
        if df.empty:
            raise ValueError(f"File '{input_path}' is empty.")
        cols = columns if columns else [df.columns[0]]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Columns {missing} not found in '{input_path}'. Available columns: {list(df.columns)}"
            )

    target_col = "_combined_text"
    df = df.copy()
    df[target_col] = build_combined_series(
        df, cols, sep=sep, include_headers=include_headers
    )
    df = df[df[target_col].str.len() >= min_len]
    return df, cols, target_col


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


def run_labeling(
    input_path: str,
    columns: list[str] | None = None,
    sep: str = "\n",
    include_headers: bool = False,
    num_samples: int = 850,
    output_path: str = "youtube_silver_labels.jsonl",
    file_mode: Literal["a", "w"] = "a",
    domain: str = "creator",
    model: str = "gpt-4o-mini",
    stride: int = 20,
    min_len: int = 3,
) -> None:
    df, cols, target_col = load_input_dataframe(
        input_path,
        columns=columns,
        sep=sep,
        include_headers=include_headers,
        min_len=min_len,
    )
    count = 0

    with open(output_path, file_mode, encoding="utf-8") as f:
        for orig_idx, row in df.iterrows():
            if stride > 1 and int(orig_idx) % stride != 0:
                continue

            text = row[target_col].strip()
            if not text:
                continue

            labels = label_text(text, domain=domain, model=model)
            record = {
                "source": input_path,
                "columns": cols,
                "domain": domain,
                "row_index": int(orig_idx),
                "text": text,
                **labels,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            count += 1
            preview = text.replace("\n", " ")[:60]
            print(f"[{count}/{num_samples}] ({file_mode}) {record['primary_category']} -> {preview}")
            if count >= num_samples:
                break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label feedback rows from one or more columns of a .csv, .tsv, or .txt file into JSONL."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to input .csv, .tsv, or .txt file",
    )
    parser.add_argument(
        "-c",
        "--columns",
        nargs="+",
        default=None,
        help="One or more column names to combine (e.g. -c review_headline review_body star_rating review_id)",
    )
    parser.add_argument(
        "--sep",
        default="\n",
        help="Separator used when joining multiple columns (default: newline)",
    )
    parser.add_argument(
        "--headers",
        action="store_true",
        help="Prefix each non-empty field with 'column_name: ' in the combined text",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["a", "w"],
        default="a",
        help="Output file mode: 'a' to append (default) or 'w' to overwrite",
    )
    parser.add_argument(
        "-n",
        "--samples",
        type=int,
        default=850,
        help="Maximum number of samples to label (default: 850)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="youtube_silver_labels.jsonl",
        help="Output JSONL path (default: youtube_silver_labels.jsonl)",
    )
    parser.add_argument(
        "--domain",
        default="creator",
        help="Domain context passed in prompt (default: creator)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model name (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=20,
        help="Process every Nth row based on original index; use 1 to process sequentially (default: 20)",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=3,
        help="Minimum character length for combined text (default: 3)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Please set OPENAI_API_KEY before running.")

    args = parse_args()
    run_labeling(
        input_path=args.input,
        columns=args.columns,
        sep=args.sep,
        include_headers=args.headers,
        num_samples=args.samples,
        output_path=args.output,
        file_mode=args.mode,
        domain=args.domain,
        model=args.model,
        stride=args.stride,
        min_len=args.min_len,
    )