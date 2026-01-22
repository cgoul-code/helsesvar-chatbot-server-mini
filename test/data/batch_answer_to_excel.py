# This script batches 30 questions through your agent and saves results to Excel.
# It assumes your project has the same modules and types as in your codebase:
# - answer_utils.get_answer_with_related_queries_as_stream
# - QuerySettings, ServerSettings, VectorIndexStore
# Adjust VECTOR_INDEX and paths as needed.
#
# Files created:
# - /mnt/data/batch_answer_to_excel.py  (the script)
# - /mnt/data/sample_questions.json     (example input JSON)
# - When you run the script, it will produce /mnt/data/answers.xlsx

import json
import os
import time
from textwrap import dedent
import logging
from dotenv import load_dotenv, find_dotenv

script_path = "/test/data/batch_answer_to_excel.py"
questions_path = "/test/data/sample_questions.json"



import asyncio
import argparse
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from llama_index.core import StorageContext, load_index_from_storage
from langchain_openai import AzureChatOpenAI

# --- Import your project utilities (these must exist in your environment) ---
from config import server_settings
from answer_utils import get_answer_as_stream
from query_utils import QuerySettings
from config import ServerSettings, VectorIndexStore




def RunningLocally():
    if 'WEBSITE_SITE_NAME' in os.environ or 'FUNCTIONS_WORKER_RUNTIME' in os.environ:
        return False
    else:
        print("Logging info locally")
        return True
    

VECTOR_INDEX_MAP = [
    {
        "name": "hvaerinnafor",
        "storage": ("." if RunningLocally() else "") + "/blobstorage/chatbot/hvaerinnafor",
        "description": "Forelskelse",
    },
    {
        "name": "hvaerinnafor_qa_bank",
        "storage": ("." if RunningLocally() else "") + "/blobstorage/chatbot/hvaerinnafor_qa_bank",
        "description": "Relaterte spørsmål",
    },
]
load_dotenv(find_dotenv())
# module-level store
vector_store = VectorIndexStore()
server_settings = ServerSettings()

LLMGPT4 = AzureChatOpenAI(
    azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    timeout=120,
    temperature = 0.0, 
    verbose=True,
)
server_settings.set_llm(LLMGPT4)

def _today_oslo_date_str() -> str:
    return datetime.now(ZoneInfo("Europe/Oslo")).date().isoformat()

def _normalize_existing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tries to map common legacy column names to the required names:
      question, expected answer
    """
    col_map = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc == "question":
            col_map[c] = "question"
        elif lc == "expected answer":
            col_map[c] = "expected answer"
        elif lc == "expected_answer":
            col_map[c] = "expected answer"
        elif lc == "questions" or lc == "question ":
            col_map[c] = "question"

    if col_map:
        df = df.rename(columns=col_map)
    return df

def _ensure_unique_answer_col(df: pd.DataFrame, base_col: str) -> str:
    """
    If the run-date column already exists, make it unique by appending time.
    """
    if base_col not in df.columns:
        return base_col

    # Same date already exists; append HH:MM:SS to avoid overwriting
    t = datetime.now(ZoneInfo("Europe/Oslo")).strftime("%H:%M:%S")
    return f"{base_col} {t}"

def upsert_answers_excel(
    out_xlsx: str,
    run_df: pd.DataFrame,
) -> None:
    """
    run_df must contain:
      - 'question'
      - 'expected answer'
      - '<run_answer_col>'
    Creates file if missing; otherwise appends new column and updates/appends rows by question.
    """
    if not os.path.exists(out_xlsx):
        run_df.to_excel(out_xlsx, index=False)
        print(f"Created Excel and saved {len(run_df)} rows to {out_xlsx}")
        return

    existing = pd.read_excel(out_xlsx)
    existing = _normalize_existing_columns(existing)

    if "question" not in existing.columns:
        # If existing file is not in expected format, fall back to preserving it as-is by
        # adding the new sheet-like structure in the same sheet.
        # (But we still try to behave sensibly.)
        existing["question"] = ""

    if "expected answer" not in existing.columns:
        existing["expected answer"] = ""

    # Identify the run's answer column (the one that's not the two base columns)
    run_answer_cols = [c for c in run_df.columns if c not in ("question", "expected answer")]
    if len(run_answer_cols) != 1:
        raise ValueError(f"run_df must have exactly 1 run answer column, found: {run_answer_cols}")
    run_answer_col = run_answer_cols[0]

    # Ensure we don't overwrite an existing same-name column
    unique_answer_col = _ensure_unique_answer_col(existing, run_answer_col)
    if unique_answer_col != run_answer_col:
        run_df = run_df.rename(columns={run_answer_col: unique_answer_col})
        run_answer_col = unique_answer_col

    # Ensure the new column exists on existing
    if run_answer_col not in existing.columns:
        existing[run_answer_col] = ""

    # Index by question for upsert
    existing_idx = existing.set_index("question", drop=False)
    run_idx = run_df.set_index("question", drop=False)

    # Update existing rows / append new ones
    for q, row in run_idx.iterrows():
        if q in existing_idx.index:
            # Fill expected answer if blank
            if (
                str(existing_idx.at[q, "expected answer"]).strip() == ""
                and str(row.get("expected answer", "")).strip() != ""
            ):
                existing_idx.at[q, "expected answer"] = row.get("expected answer", "")

            existing_idx.at[q, run_answer_col] = row.get(run_answer_col, "")
        else:
            # Append new row with all columns existing currently
            new_row = {col: "" for col in existing_idx.columns}
            new_row["question"] = row.get("question", "")
            new_row["expected answer"] = row.get("expected answer", "")
            new_row[run_answer_col] = row.get(run_answer_col, "")
            existing_idx = pd.concat([existing_idx, pd.DataFrame([new_row]).set_index("question", drop=False)])

    out_df = existing_idx.reset_index(drop=True)
    out_df.to_excel(out_xlsx, index=False)
    print(f"Updated Excel (added '{run_answer_col}') and saved to {out_xlsx}")


def _extract_answer_and_refs(answer_stream_text: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    The stream sends 'Answer' chunks that include sections like:
      "## Du spurte\n..."
      "## Svar\n..."
      "## Referanser\n- [Title](URL)\n- ..."
    We'll extract the main answer between '## Svar' and '## Referanser', and parse the references.
    If markers aren't found, we fall back to returning the entire stream text as the answer.
    """
    ans = answer_stream_text

    # Normalize newlines
    ans = ans.replace("\r\n", "\n").replace("\r", "\n")

    # Find sections
    svar_start = ans.find("## Svar")
    refs_start = ans.find("## Referanser")

    if svar_start != -1:
        content_after_svar = ans[svar_start + len("## Svar"):]
        if refs_start != -1:
            main_answer = content_after_svar[: refs_start - (svar_start + len("## Svar"))].strip()
        else:
            main_answer = content_after_svar.strip()
    else:
        # Fallback: return all text
        main_answer = ans.strip()

    references: List[Dict[str, str]] = []
    if refs_start != -1:
        refs_block = ans[refs_start + len("## Referanser"):]
        # lines like "- [Title](URL)"
        for line in refs_block.splitlines():
            line = line.strip()
            if not line.startswith("- "):
                continue
            m = re.match(r"- \[(?P<title>.+?)\]\((?P<url>.+?)\)", line)
            if m:
                references.append({"title": m.group("title"), "url": m.group("url")})

    return main_answer, references


async def _run_one_question(
    question: str,
    vector_index: str,
    similarity_top_k: int = 5,
    similarity_cutoff: float = 0.75,
    related_only: bool = False,
    main_category: Optional[str] = None,
    query_severity: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Runs your agent for a single user question and returns a dict with parsed results.
    """
    global vector_store
    global server_settings
    

    qs = QuerySettings(
        user_content=question,
        vectorIndex=vector_index,
        response_mode="tree_summarize",   # must be accepted by your codebase
        similarity_top_k=similarity_top_k,
        similarity_cutoff=similarity_cutoff,
        related_only=related_only,
        main_category=main_category,
        query_severity=query_severity,
    )
    print(qs)

    # Collect stream
    answer_buffer: List[str] = []
    related_queries_json: Optional[str] = None
    refined_query_text: Optional[str] = None
    maincategory_text: Optional[str] = None
    subcategories_text: Optional[str] = None

    async for chunk in get_answer_as_stream(qs, server_settings, vector_store):
        # Each chunk is expected to be a dict like {"event": "...", "structured_answer_delta": "..."}
        try:
            event = chunk.get("event")
        except AttributeError:
            event = None

        # Text payload field can vary; try known keys in priority order
        text_delta = None
        for k in ("structured_answer_delta", "delta", "text", "message", "content"):
            if isinstance(chunk, dict) and k in chunk and isinstance(chunk[k], str):
                text_delta = chunk[k]
                break

        if event == "answer" and text_delta:
            answer_buffer.append(text_delta)
        elif event == "related queries" and text_delta:
            related_queries_json = text_delta  # raw JSON array string
        elif event == "refined query" and text_delta:
            refined_query_text = text_delta
        elif event == "Maincategory" and text_delta:
            maincategory_text = text_delta
        elif event == "Subcategories" and text_delta:
            subcategories_text = text_delta
        # ignore other events for the Excel export

    answer_stream_text = "".join(answer_buffer)
    main_answer, references = _extract_answer_and_refs(answer_stream_text)

    return {
        "question": question,
        "refined_query": refined_query_text,
        "main_category": maincategory_text,
        "subcategories": subcategories_text,
        "answer_markdown": main_answer,
        "references": references,
        "related_queries_json": related_queries_json,
        "raw_stream": answer_stream_text,
    }
async def run_batch(
    questions: List[Dict[str, Any]],   # list of dicts from load_questions()
    vector_index: str,
    out_xlsx: str,
    input_json_path: str,              # NEW: path to write updated JSON back
    similarity_top_k: int = 10,
    similarity_cutoff: float = 0.75,
    related_only: bool = False,
    main_category: Optional[str] = None,
    query_severity: Optional[str] = None,
):
    rows: List[Dict[str, Any]] = []

    run_date = _today_oslo_date_str()
    answer_col = f"answer - run date {run_date}"

    # Helper to check blank values robustly
    def _is_blank(s: Any) -> bool:
        return s is None or str(s).strip() == ""

    for i, item in enumerate(questions, 1):
        q = item["question"]
        exp = item.get("expected_answer", "")

        print(f"[{i}/{len(questions)}] Running agent for: {q[:80]}{'...' if len(q) > 80 else ''}")

        try:
            res = await _run_one_question(
                q,
                vector_index=vector_index,
                similarity_top_k=similarity_top_k,
                similarity_cutoff=similarity_cutoff,
                related_only=related_only,
                main_category=main_category,
                query_severity=query_severity,
            )
            answer_text = res["answer_markdown"]
        except Exception as e:
            answer_text = f"ERROR: {e}"

        # If expected_answer is blank in input, fill it with the produced answer
        if _is_blank(exp):
            item["expected_answer"] = answer_text
            exp = answer_text

        # Excel wants column name "expected answer" (space)
        rows.append(
            {
                "question": q,
                "expected answer": exp,
                answer_col: answer_text,
            }
        )

    # 1) Write/append Excel (creates file if missing, otherwise adds new run column)
    run_df = pd.DataFrame(rows, columns=["question", "expected answer", answer_col])
    upsert_answers_excel(out_xlsx, run_df)

    # 2) Write the updated JSON back (now includes filled expected_answer)
    with open(input_json_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"Updated input JSON with filled expected_answer where missing: {input_json_path}")

def load_questions(json_path: str) -> List[Dict[str, Any]]:
    """
    Supports:
      - ["q1", "q2", ...]  -> converted to [{"question": "...", "expected_answer": ""}, ...]
      - [{"question":"...", "expected_answer":"..."}, ...]
      - [{"question":"...", "expected answer":"..."}, ...]  (will be normalized to expected_answer)
    Returns list of dicts (original objects preserved as much as possible).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON must be a list")

    # list[str] -> list[dict]
    if all(isinstance(x, str) for x in data):
        return [{"question": q, "expected_answer": ""} for q in data]

    # list[dict]
    if all(isinstance(x, dict) and "question" in x for x in data):
        normalized: List[Dict[str, Any]] = []
        for x in data:
            # preserve other keys in x
            item = dict(x)
            if "expected_answer" not in item and "expected answer" in item:
                item["expected_answer"] = item.get("expected answer", "")
                # optionally remove the spaced version to avoid duplicates
                item.pop("expected answer", None)

            if "expected_answer" not in item:
                item["expected_answer"] = ""

            # ensure strings
            item["question"] = str(item["question"])
            if item["expected_answer"] is None:
                item["expected_answer"] = ""
            else:
                item["expected_answer"] = str(item["expected_answer"])

            normalized.append(item)
        return normalized

    raise ValueError("JSON must be a list of strings or list of objects with a 'question' field.")



def read_all_indexes_from_storage(vector_map):
    """Load all indexes into the module-level vector_store."""
    global vector_store
    found_any = False

    # Do NOT re-instantiate vector_store here; we fill the existing one.
    for item in vector_map:
        start = time.time()
        name = item['name']
        storage = item['storage']
        desc = item['description']
        logging.info("-------------------------------")

        if os.path.exists(storage):
            logging.info(f"Loading index '{name}' from {storage}")
            storage_ctx = StorageContext.from_defaults(persist_dir=storage)
            idx = load_index_from_storage(storage_ctx)
            vector_store.add(name, idx, desc)
            found_any = True
        else:
            logging.warning(f"Index directory not found: {storage}")

        elapsed = time.time() - start
        logging.info(f"Time taken for {name}: {elapsed:.2f}s")

    vector_store.indexes_loaded = found_any
    return found_any

logging.basicConfig(
    level=logging.INFO,
    force=True
)

def main():
    
   
    
    # Load indexes
    try:
        found_any = read_all_indexes_from_storage(VECTOR_INDEX_MAP)
        if found_any:
            logging.info("Indexes successfully read from storage.")
        else:
            logging.info("Indexes not successfully read from storage.")
    except Exception as e:
        logging.error(f"Failed to read indexes from storage: {e}")
        vector_store.indexes_loaded = False
        
    # --- Use the store ---
    # NOTE: correct the name here ("hvaerinnafor", not "hvaerinnfor")
    entry = vector_store.get("hvaerinnafor")
    if entry is None:
        raise RuntimeError("Index 'hvaerinnafor' not found. Loaded: "
                        + ", ".join([e.name for e in vector_store.get_all()]))

    index = entry.index  # type: ignore
    vector_index_description = entry.description
    logging.info("Found entry: %s", vector_index_description)
    
    ap = argparse.ArgumentParser(description="Batch 30 questions through your agent and export to Excel.")
    ap.add_argument("--input", "-i", required=True, help="Path to JSON file with questions")
    ap.add_argument("--out", "-o", default="answers.xlsx", help="Output Excel path")
    ap.add_argument("--vector-index", "-v", required=True, help="Name of the vector index to use")
    ap.add_argument("--topk", type=int, default=5, help="similarity_top_k")
    ap.add_argument("--cutoff", type=float, default=0.35, help="similarity_cutoff")
    ap.add_argument("--related-only", action="store_true", help="Only compute related queries/categories (no answers)")
    ap.add_argument("--main-category", default=None, help="Force a main category (optional)")
    ap.add_argument("--query-severity", default=None, help="Force a severity: Green|Yellow|Red (optional)")
    args = ap.parse_args()

    questions = load_questions(args.input)

    print(f"WARNING: Input contains {len(questions)} questions (not 30). Proceeding.")

    asyncio.run(
        run_batch(
            questions=questions,
            vector_index=args.vector_index,
            out_xlsx=args.out,
            input_json_path=args.input,   # NEW
            similarity_top_k=args.topk,
            similarity_cutoff=args.cutoff,
            related_only=args.related_only,
            main_category=args.main_category,
            query_severity=args.query_severity,
        )
    )


if __name__ == "__main__":
    main()
