import json
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.config import get_stream_writer
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator


class Reference(TypedDict):
    name: str
    url: str
    icon_url: str
    relevancy_index: float


def _emit(delta: str, event: str = "systeminfo") -> None:
    writer = get_stream_writer()
    writer({"event": event, "structured_answer_delta": delta})


def _as_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default


def _as_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return default


def _node_text(n: Any) -> str:
    t = getattr(n, "text", None)
    if t:
        return t

    get_content = getattr(n, "get_content", None)
    if callable(get_content):
        return get_content(metadata_mode="all") or ""

    return getattr(n, "get_text", lambda: "")() or ""


def _build_related_queries_retriever(
    index_qa_bank: VectorStoreIndex,
    *,
    top_k: int,
    cutoff: float,
    query_severity: Optional[str],
    main_category: Optional[str],
) -> BaseRetriever:
    top_k = _as_int(top_k, 5) or 5
    cutoff = _as_float(cutoff, 0.0) or 0.0

    if query_severity == "Green":
        allowed_sev = ["Green"]
    elif query_severity == "Yellow":
        allowed_sev = ["Green", "Yellow"]
    else:
        allowed_sev = ["Green", "Yellow", "Red"]

    filters_list: List[MetadataFilter] = [
        MetadataFilter(key="valid", value=1, operator=FilterOperator.EQ),
        MetadataFilter(key="severity", value=allowed_sev, operator=FilterOperator.IN),
    ]

    if main_category:
        filters_list.append(
            MetadataFilter(key="category", value=main_category, operator=FilterOperator.EQ)
        )

    composite = MetadataFilters(filters=filters_list, condition="and")
    return index_qa_bank.as_retriever(
        similarity_top_k=top_k,
        similarity_cutoff=cutoff,
        filters=composite,
    )


def _dedupe_references(refs: List[Reference], top_k: int = 5) -> List[Reference]:
    if not refs:
        return []
    seen = set()
    out: List[Reference] = []
    for r in refs:
        url = r.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(r)
        if len(out) >= top_k:
            break
    return out
