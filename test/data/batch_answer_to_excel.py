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
# Reuse the canonical VECTOR_INDEX_MAP from config.py so this script picks up
# every registered index (hvaerinnafor, hvaerinnafor_qa_bank, psa_ssa_topics,
# hvaerinnafor_unified) without drift.
from config import server_settings, ServerSettings, VectorIndexStore, VECTOR_INDEX_MAP
from answer_utils import get_answer_as_stream
from query_utils import QuerySettings


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
      - one or more per-run columns (e.g. 'answer - run date YYYY-MM-DD',
        'references - run date YYYY-MM-DD').
    Creates file if missing; otherwise appends the new per-run columns and
    updates/appends rows by question. When any per-run column collides with an
    existing column, the SAME HH:MM:SS suffix is applied to every per-run
    column so they remain paired.
    """
    if not os.path.exists(out_xlsx):
        run_df.to_excel(out_xlsx, index=False)
        print(f"Created Excel and saved {len(run_df)} rows to {out_xlsx}")
        return

    existing = pd.read_excel(out_xlsx)
    existing = _normalize_existing_columns(existing)

    if "question" not in existing.columns:
        existing["question"] = ""
    if "expected answer" not in existing.columns:
        existing["expected answer"] = ""

    base_cols = ("question", "expected answer")
    run_cols = [c for c in run_df.columns if c not in base_cols]
    if not run_cols:
        raise ValueError("run_df must contain at least one per-run column besides the base columns.")

    # If ANY per-run column collides with an existing column, suffix them all
    # with the same HH:MM:SS so they stay paired in the output.
    if any(c in existing.columns for c in run_cols):
        suffix = datetime.now(ZoneInfo("Europe/Oslo")).strftime("%H:%M:%S")
        rename_map = {c: f"{c} {suffix}" for c in run_cols}
        run_df = run_df.rename(columns=rename_map)
        run_cols = list(rename_map.values())

    for c in run_cols:
        if c not in existing.columns:
            existing[c] = ""

    # Index by question for upsert
    existing_idx = existing.set_index("question", drop=False)
    run_idx = run_df.set_index("question", drop=False)

    for q, row in run_idx.iterrows():
        if q in existing_idx.index:
            if (
                str(existing_idx.at[q, "expected answer"]).strip() == ""
                and str(row.get("expected answer", "")).strip() != ""
            ):
                existing_idx.at[q, "expected answer"] = row.get("expected answer", "")
            for c in run_cols:
                existing_idx.at[q, c] = row.get(c, "")
        else:
            new_row = {col: "" for col in existing_idx.columns}
            new_row["question"] = row.get("question", "")
            new_row["expected answer"] = row.get("expected answer", "")
            for c in run_cols:
                new_row[c] = row.get(c, "")
            existing_idx = pd.concat(
                [existing_idx, pd.DataFrame([new_row]).set_index("question", drop=False)]
            )

    out_df = existing_idx.reset_index(drop=True)
    out_df.to_excel(out_xlsx, index=False)
    print(f"Updated Excel (added {run_cols}) and saved to {out_xlsx}")


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
    psa_ssa_threshold: float = 0.65,
    qa_bank_index: Optional[str] = None,
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
        psa_ssa_threshold=psa_ssa_threshold,
        qa_bank_index=qa_bank_index,
    )
    print(qs)

    # Collect stream
    answer_buffer: List[str] = []
    references_lines: List[str] = []
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
        elif event == "references" and text_delta:
            references_lines.append(text_delta)
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
    main_answer, md_references = _extract_answer_and_refs(answer_stream_text)

    # Prefer references streamed as `references` events (authoritative — emitted
    # directly from the retrieved nodes in emit_query_answer_references). Fall
    # back to references parsed from the answer markdown only if the stream
    # didn't emit any.
    streamed_refs = _parse_streamed_references(references_lines)
    references = streamed_refs if streamed_refs else md_references

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


_REF_LINE_RE = re.compile(
    r"^\[(?P<name>.+?)\]\((?P<url>.+?)\)(?:\s*\|\|IMG\|\|\s*(?P<icon>.+?))?\s*$"
)


def _parse_streamed_references(lines: List[str]) -> List[Dict[str, str]]:
    """Parse references emitted as one bullet per `references` event.

    Each line is `[Name](url) ||IMG|| icon_url\\n` or `[Name](url)\\n` —
    see emit_query_answer_references in agent_workflow_answer.py.
    """
    out: List[Dict[str, str]] = []
    seen_urls = set()
    for raw in lines:
        for piece in raw.splitlines():
            piece = piece.strip()
            if not piece:
                continue
            m = _REF_LINE_RE.match(piece)
            if not m:
                continue
            url = m.group("url").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            out.append({
                "title": m.group("name").strip(),
                "url": url,
                "icon_url": (m.group("icon") or "").strip(),
            })
    return out


def _format_references_for_cell(refs: List[Dict[str, str]]) -> str:
    """One reference per line as 'Title — URL' for readability in Excel."""
    if not refs:
        return ""
    return "\n".join(f"{r.get('title','').strip()} — {r.get('url','').strip()}" for r in refs)
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
    psa_ssa_threshold: float = 0.65,
    qa_bank_index: Optional[str] = None,
):
    rows: List[Dict[str, Any]] = []

    run_date = _today_oslo_date_str()
    answer_col = f"answer - run date {run_date}"
    refs_col = f"references - run date {run_date}"

    # Helper to check blank values robustly
    def _is_blank(s: Any) -> bool:
        return s is None or str(s).strip() == ""

    for i, item in enumerate(questions, 1):
        q = item["question"]
        exp = item.get("expected_answer", "")

        print(f"[{i}/{len(questions)}] Running agent for: {q[:80]}{'...' if len(q) > 80 else ''}")

        refs_text = ""
        try:
            res = await _run_one_question(
                q,
                vector_index=vector_index,
                similarity_top_k=similarity_top_k,
                similarity_cutoff=similarity_cutoff,
                related_only=related_only,
                main_category=main_category,
                query_severity=query_severity,
                psa_ssa_threshold=psa_ssa_threshold,
                qa_bank_index=qa_bank_index,
            )
            answer_text = res["answer_markdown"]
            refs_text = _format_references_for_cell(res.get("references") or [])
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
                refs_col: refs_text,
            }
        )

    # 1) Write/append Excel (creates file if missing, otherwise adds new run columns)
    run_df = pd.DataFrame(rows, columns=["question", "expected answer", answer_col, refs_col])
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
    ap = argparse.ArgumentParser(description="Batch questions through your agent and export to Excel.")
    ap.add_argument("--input", "-i", required=True, help="Path to JSON file with questions")
    ap.add_argument("--out", "-o", default="answers.xlsx", help="Output Excel path")
    ap.add_argument("--vector-index", "-v", required=True,
                    help="Main vector index to use (e.g. hvaerinnafor, hvaerinnafor_unified)")
    ap.add_argument("--topk", type=int, default=5, help="similarity_top_k")
    ap.add_argument("--cutoff", type=float, default=0.75, help="similarity_cutoff")
    ap.add_argument("--related-only", action="store_true", help="Only compute related queries/categories (no answers)")
    ap.add_argument("--main-category", default=None, help="Force a main category (optional)")
    ap.add_argument("--query-severity", default=None, help="Force a severity: Green|Yellow|Red (optional)")
    ap.add_argument("--psa-ssa-threshold", type=float, default=0.65,
                    help="Cutoff for psa_ssa_topics tier-1 route. Set to 0 to disable (e.g. for pure unified mode).")
    ap.add_argument("--qa-bank-index", default=None,
                    help="Explicit QA-bank index name. If omitted the server falls back to "
                         "'{vectorIndex}_qa_bank' then 'hvaerinnafor_qa_bank'. "
                         "Set to 'hvaerinnafor_qa_bank' when --vector-index is hvaerinnafor_unified "
                         "to silence the 'QA-bank not found' warning.")
    args = ap.parse_args()

    # Load every index defined in config.VECTOR_INDEX_MAP so the cascade
    # (psa_ssa_topics) and the unified index are available alongside the
    # legacy article + qa_bank pair.
    try:
        found_any = read_all_indexes_from_storage(VECTOR_INDEX_MAP)
        if found_any:
            logging.info("Indexes successfully read from storage.")
        else:
            logging.info("Indexes not successfully read from storage.")
    except Exception as e:
        logging.error(f"Failed to read indexes from storage: {e}")
        vector_store.indexes_loaded = False

    # Fail fast if the requested main index didn't load.
    entry = vector_store.get(args.vector_index)
    if entry is None:
        raise RuntimeError(
            f"Index '{args.vector_index}' not found. Loaded: "
            + ", ".join([e.name for e in vector_store.get_all()])
        )
    logging.info("Using main index: %s — %s", entry.name, entry.description)

    questions = load_questions(args.input)

    print(f"Input contains {len(questions)} questions. Proceeding.")

    asyncio.run(
        run_batch(
            questions=questions,
            vector_index=args.vector_index,
            out_xlsx=args.out,
            input_json_path=args.input,
            similarity_top_k=args.topk,
            similarity_cutoff=args.cutoff,
            related_only=args.related_only,
            main_category=args.main_category,
            query_severity=args.query_severity,
            psa_ssa_threshold=args.psa_ssa_threshold,
            qa_bank_index=args.qa_bank_index,
        )
    )


if __name__ == "__main__":
    main()
