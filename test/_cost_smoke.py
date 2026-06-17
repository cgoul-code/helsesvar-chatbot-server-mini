"""Ad-hoc smoke for the per-model token/cost accounting. Not a fixture.

Runs one question through the full agent and prints every query_status event
(there should be TWO: the answer-phase one, then a re-emit from related_queries
folding in the full cost) plus the final token/cost breakdown.
"""
from __future__ import annotations

import asyncio
import json
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


def _load_indexes() -> None:
    for item in VECTOR_INDEX_MAP:
        name, storage, desc = item["name"], item["storage"], item["description"]
        if os.path.exists(storage):
            ctx = StorageContext.from_defaults(persist_dir=storage)
            vector_store.add(name, load_index_from_storage(ctx), desc)
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
    statuses = []
    async for chunk in get_answer_as_stream(qs, server_settings, vector_store):
        if not isinstance(chunk, dict):
            continue
        if chunk.get("event") != "query_status":
            continue
        for k in ("structured_answer_delta", "delta", "text", "message", "content"):
            v = chunk.get(k)
            if isinstance(v, str):
                try:
                    statuses.append(json.loads(v))
                except json.JSONDecodeError:
                    pass
                break

    print("\n" + "=" * 70)
    print(f"Q: {q}")
    print(f"  query_status events: {len(statuses)} (expect 2: answer-phase + re-emit)")
    for i, s in enumerate(statuses):
        print(
            f"  [{i}] stance={s.get('stance')!r} "
            f"in={s.get('input_tokens')} out={s.get('output_tokens')} "
            f"usd={s.get('cost_usd')} nok={s.get('cost_nok')}"
        )
    if statuses:
        final = statuses[-1]
        # Re-derive cost from the published per-model prices to cross-check.
        pin = float(os.getenv("PRICE_INPUT_USD_PER_M", "0") or 0)
        pout = float(os.getenv("PRICE_OUTPUT_USD_PER_M", "0") or 0)
        print(
            f"  FINAL: total in/out = {final.get('input_tokens')}/{final.get('output_tokens')}, "
            f"cost_nok = {final.get('cost_nok')}"
        )
        print(f"  (main prices in/out = {pin}/{pout} USD per 1M; fast priced separately)")


async def _main() -> None:
    print("Loading indexes (~30-60s)...")
    t = time.time()
    _load_indexes()
    print(f"  loaded in {time.time() - t:.1f}s")
    await _run_one("Hvilke prevensjonsmidler finnes?")


if __name__ == "__main__":
    asyncio.run(_main())
