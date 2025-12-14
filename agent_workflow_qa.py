import json
import logging

from operator import add
from typing import Any, Dict, List, Literal, Optional, Tuple

from typing_extensions import TypedDict



from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator

from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer
from agent_shared import Reference, _emit, _node_text, _build_related_queries_retriever, _as_int, _as_float, _dedupe_references



# ---------------------------------------------------------
# Datamodeller og typer
# ---------------------------------------------------------

class Reference(TypedDict):
    name: str
    url: str
    icon_url: str
    relevancy_index: float
    
    
class State_Related(TypedDict):
    # infra
    llm: Any
    index: VectorStoreIndex # Text-bank index
    index_related_queries: VectorStoreIndex # QA-bank index
 
    categories: List[Dict[str, Any]]

    # query context
    query: str


    # how we got here
    from_related_q: bool        # True if user clicked a related Q
    from_node_id: str           # node id of the QA-bank entry when from_related_q=True

    # similarity settings for related queries
    similarity_cutoff: float
    similarity_top_k: int

    # outputs
    main_category: str
    query_severity: Literal["Green", "Yellow", "Red", ""]
    references: List[Reference]
    final_answer: str



def _fetch_answer_from_related_question(
    node_id: str,
    index: Optional[VectorStoreIndex] = None,
    index_qa: Optional[VectorStoreIndex] = None,

    ) -> Optional[Tuple[str, List[Reference], str, str]]:
    """
    Look up a Q→A hit for `query`. If the top-scored node has an 'answer' in metadata
    and its score >= `score_threshold`, return (answer, refs). Otherwise return None.
    """

    
    ds_qa = index_qa.storage_context.docstore    
    qa_node = ds_qa.get_node(node_id)
    
    meta = getattr(qa_node, "metadata", {}) or {}
    answer = meta.get("answer", "")
    from_doc_id = meta.get("from_doc_id", "")
    
    if not answer:
        return None
    
    title = ""
    url = meta.get("url", "")
    icon_url = ""
    category = meta.get("category", "")
    severity = meta.get("severity", "Green")
    
    if index is not None and from_doc_id:
        ds = index.storage_context.docstore
        try:
            src_doc = ds.get_document(from_doc_id)  # <-- THIS is the key
            text_meta = getattr(src_doc, "metadata", {}) or {}
            title = text_meta.get("title", title)
            url = text_meta.get("url", url)
            icon_url = text_meta.get("icon_url", icon_url)
            category = text_meta.get("category", category)
            severity = text_meta.get("severity", severity)
        except Exception as e:
            logging.warning(f"Could not fetch document {from_doc_id}: {e}")


    refs: List[Reference] = [
        {
            "name": title or url or "Uten tittel",
            "url": url,
            "icon_url": icon_url,
            "relevancy_index": 0.0,
        }
    ]

    return answer, refs, category, severity


def get_metadata_from_node_id(state: State_Related):
    """
    If from_related_q is True, try to answer directly from the Q→A index.
    On hit -> return final_answer/references and route='emit'.
    Otherwise route to related_only or full.
    """
    try:
        idx_qa = state.get("index_related_queries")
        idx = state.get("index")
        
        result = _fetch_answer_from_related_question(
            node_id=state["from_node_id"],
            index = idx,
            index_qa=idx_qa,
        )
        if result:
            answer, refs, category, severity = result
            # IMPORTANT: return the updates; do not mutate `state`
            return {
                "final_answer": answer,
                "references": refs,
                "main_category": category,
                "query_severity": severity,
            }

        # no fast hit; decide normal route
        return {}

    except Exception as e:
        logging.error(f"maybe_use_related_q error: {e}")
        return {}
        
        
def emit_query_answer_references(state: State_Related) -> Dict[str, Any]:
    """Emitter endelig svar + referanser som events."""
    
    _emit( "Aggregating the final answer", event="info")

    q = state.get("query")
    if q:
        _emit(q, event="Refined query")

    _emit(f"\n## Du spurte\n{state['query']}\n")
    _emit("\n## Svar\n")

    answer = state["final_answer"]

    for line in answer.splitlines(True):
        _emit(line, event = "answer")
    _emit("\n", event = "answer")

    top5 = state["references"]

    if top5:
        #_emit("\n## Referanser\n", event = 'references')
        for r in top5:
            name = r.get("name", "Uten tittel")
            url = r.get("url", "#")
            icon_url = r.get("icon_url")

            if icon_url:
                bullet = f'- [{name}]({url}) ![]({icon_url})\n'
            else:
                bullet = f'- [{name}]({url})\n'
            _emit(bullet, event="references")
   

    return {}

def related_queries(state: State_Related) -> dict:

    writer = get_stream_writer()
    related_queries = []

    try:        
        # assume you have index_qa_bank in state (or re-access via your vector_store)
        # uses main_category to retrieve queries relevant for refined_query
        retriever = _build_related_queries_retriever(
            index_qa_bank=state["index_related_queries"],            # ensure you stored this in state earlier
            top_k=state["similarity_top_k"],
            cutoff=state["similarity_cutoff"],
            query_severity=state.get("query_severity"),      # may be None → defaults to all
            main_category=state.get("main_category"),        # may be None → ignored
        )

        results = retriever.retrieve(state["query"])

        if results:
            def score_or_min(r):
                return float("-inf") if (getattr(r, "score", None) is None) else r.score

            max_idx = max(range(len(results)), key=lambda i: score_or_min(results[i]))
            max_score = score_or_min(results[max_idx])
            

            if max_score > 0.7:
                logging.info(f"Dropping top candidate with score {max_score:.3f} (> 0.7)")
                results = [r for i, r in enumerate(results) if i != max_idx]        

        candidates = []
        for r in results[:5]:
            node = getattr(r, "node", r)  # NodeWithScore -> TextNode
            meta = getattr(node, "metadata", {}) or {}
            text = _node_text(node)
            # Use your helper to collect possible ids (metadata ids + id_/node_id)
   
            node_id = getattr(node, "node_id", getattr(node, "id_", ""))

            sev = meta.get("severity", "")
            cat = meta.get("category", "")
            # Prefer from_doc_id, fallback to node_id / id_
            doc_id = meta.get("from_doc_id", getattr(node, "node_id", getattr(node, "id_", "")))
            score = float(getattr(r, "score", 0.0) or 0.0)
            
            s = {"id": str(doc_id),
                "node_id": node_id,
                "text": text,
                "severity": sev}
            
            candidates.append({
                "id": str(doc_id),
                "node_id": node_id,
                "text": text,
                "severity": sev,
                # keep if you want to inspect:
                # "score": score,
                # "category": cat,
            })
    
        related_queries = [{"keyword": s.get("severity", ""), "query": s.get("text",""), "node_id": s.get("node_id", "")} for s in candidates]

        related_queries_payload = json.dumps(related_queries, ensure_ascii=False)

        # Emit ONLY the JSON array (client listens for this event)
        _emit(related_queries_payload, event="related queries")
                                           
        
    except Exception as e:
       logging.error(f"Failed to execute agent: {e} ")       
        
    return {}

# ---------------------------------------------------------
# Bygg workflow
# ---------------------------------------------------------

builder = StateGraph(State_Related)

builder.add_node("get_metadata_from_node_id", get_metadata_from_node_id)
builder.add_node("emit_query_answer_references", emit_query_answer_references)
builder.add_node("related_queries", related_queries)


builder.add_edge(START, "get_metadata_from_node_id")
builder.add_edge("get_metadata_from_node_id", "emit_query_answer_references")
builder.add_edge("emit_query_answer_references", "related_queries")
builder.add_edge("related_queries", END)

related_qa_workflow = builder.compile()