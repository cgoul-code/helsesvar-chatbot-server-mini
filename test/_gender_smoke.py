"""Ad-hoc smoke test for the asker_gender feature. Not a fixture.

Runs a few questions through the full agent and prints the query_status
(stance/severity/asker_gender) plus the final answer, so we can eyeball
whether the perspective/gender handling is correct.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time

from dotenv import find_dotenv, load_dotenv
from llama_index.core import StorageContext, load_index_from_storage

from config import VECTOR_INDEX_MAP, ServerSettings, VectorIndexStore
from llm_provider import build_chat_llm, build_fast_chat_llm
from answer_utils import get_answer_as_stream
from query_utils import QuerySettings

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

load_dotenv(find_dotenv(), override=True)

vector_store = VectorIndexStore()
server_settings = ServerSettings()
server_settings.set_llm(build_chat_llm())
server_settings.set_fast_llm(build_fast_chat_llm())

QUESTIONS = [
    "Jeg er gravid, jeg er 17 år, hva skal jeg gjøre",
    "Hvilke prevensjonsmidler finnes?",
    "Kjæresten min er gravid, hva skal jeg gjøre?",
]


def _load_indexes() -> None:
    for item in VECTOR_INDEX_MAP:
        name, storage, desc = item["name"], item["storage"], item["description"]
        if os.path.exists(storage):
            ctx = StorageContext.from_defaults(persist_dir=storage)
            vector_store.add(name, load_index_from_storage(ctx), desc)
        else:
            print(f"  (skip) index dir not found: {storage}")
    vector_store.indexes_loaded = True


async def _run_one(q: str) -> None:
    qs = QuerySettings(
        user_content=q,
        vectorIndex="hvaerinnafor_unified",
        response_mode="tree_summarize",
        similarity_top_k=5,
        similarity_cutoff=0.75,
        psa_ssa_threshold=0.0,
        qa_bank_index="hvaerinnafor_qa_bank",
        claims_valid_threshold=1.0,
        entailment_check=True,
    )
    status = {}
    answer = []
    async for chunk in get_answer_as_stream(qs, server_settings, vector_store):
        if not isinstance(chunk, dict):
            continue
        event = chunk.get("event")
        text = None
        for k in ("structured_answer_delta", "delta", "text", "message", "content"):
            v = chunk.get(k)
            if isinstance(v, str):
                text = v
                break
        if event == "query_status" and text:
            try:
                status = json.loads(text)
            except json.JSONDecodeError:
                pass
        elif event == "answer" and text:
            answer.append(text)

    print("\n" + "=" * 70)
    print(f"Q: {q}")
    print(f"  refined_query : {status.get('refined_query', '')}")
    print(f"  severity      : {status.get('query_severity', '')}")
    print(f"  stance        : {status.get('stance', '')}")
    print(f"  asker_gender  : {status.get('asker_gender', '')}")
    print(f"  validate      : {status.get('validate_response_result', '')}")
    print("  --- answer ---")
    print("".join(answer).strip() or "(no answer text)")


async def _main() -> None:
    print("Loading indexes (~30-60s)...")
    t = time.time()
    _load_indexes()
    print(f"Loaded in {time.time() - t:.1f}s")
    for q in QUESTIONS:
        await _run_one(q)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, force=True)
    asyncio.run(_main())
