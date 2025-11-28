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

import pandas as pd
from llama_index.core import StorageContext, load_index_from_storage
from langchain_openai import AzureChatOpenAI

# --- Import your project utilities (these must exist in your environment) ---
from config import server_settings
from answer_utils import get_answer_with_related_queries_as_stream
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
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    timeout=120,
    reasoning_effort="minimal",
    verbose=True
)
server_settings.set_llm(LLMGPT4)


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

    async for chunk in get_answer_with_related_queries_as_stream(qs, server_settings, vector_store):
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

        if event == "Answer" and text_delta:
            answer_buffer.append(text_delta)
        elif event == "Related queries" and text_delta:
            related_queries_json = text_delta  # raw JSON array string
        elif event == "Refined query" and text_delta:
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
    questions: List[str],
    vector_index: str,
    out_xlsx: str,
    similarity_top_k: int = 10,
    similarity_cutoff: float = 0.75,
    related_only: bool = False,
    main_category: Optional[str] = None,
    query_severity: Optional[str] = None,
):
    rows: List[Dict[str, Any]] = []

    for i, q in enumerate(questions, 1):
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
        except Exception as e:
            res = {
                "question": q,
                "answer_markdown": f"ERROR: {e}",
                "references": [],
                "refined_query": None,
                "main_category": None,
                "subcategories": None,
                "related_queries_json": None,
                "raw_stream": "",
            }
        rows.append(res)

    # Flatten references for Excel
    def refs_to_text(refs: List[Dict[str, str]]) -> str:
        if not refs:
            return ""
        return "; ".join([f"{r.get('title','')}: {r.get('url','')}" for r in refs])

    df = pd.DataFrame(
        {
            "Question": [r["question"] for r in rows],
            "RefinedQuery": [r.get("refined_query") for r in rows],
            "MainCategory": [r.get("main_category") for r in rows],
            "Subcategories": [r.get("subcategories") for r in rows],
            "AnswerMarkdown": [r["answer_markdown"] for r in rows],
            "References": [refs_to_text(r.get("references", [])) for r in rows],
            "RelatedQueriesJSON": [r.get("related_queries_json") for r in rows],
            "RawStream": [r.get("raw_stream") for r in rows],
        }
    )

    # Write to Excel
    df.to_excel(out_xlsx, index=False)
    print(f"Saved {len(rows)} rows to {out_xlsx}")


def load_questions(json_path: str) -> List[str]:
    """
    Accepts either:
      - a list of strings: ["q1", "q2", ...]
      - a list of objects with 'question' key: [{"question":"..."}, ...]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        if all(isinstance(x, str) for x in data):
            return data
        if all(isinstance(x, dict) and "question" in x for x in data):
            return [x["question"] for x in data]
    raise ValueError("JSON must be a list of strings or a list of objects with a 'question' field")


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
            similarity_top_k=args.topk,
            similarity_cutoff=args.cutoff,
            related_only=args.related_only,
            main_category=args.main_category,
            query_severity=args.query_severity,
        )
    )


if __name__ == "__main__":
    main()
