import json
import logging

from operator import add
from typing import Any, Dict, List, Literal, Optional, Tuple

from typing_extensions import TypedDict



from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever

from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer



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



def _emit(delta: str, event: str = "systeminfo") -> None:
    """Sender streaming-event til klient (UI)."""
    writer = get_stream_writer()
    writer({"event": event, "structured_answer_delta": delta})

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
        _emit(json.dumps(q, ensure_ascii=False), event="Refined query")

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

# ---------------------------------------------------------
# Bygg workflow
# ---------------------------------------------------------

builder = StateGraph(State_Related)

builder.add_node("get_metadata_from_node_id", get_metadata_from_node_id)
builder.add_node("emit_query_answer_references", emit_query_answer_references)

builder.add_edge(START, "get_metadata_from_node_id")
builder.add_edge("get_metadata_from_node_id", "emit_query_answer_references")
builder.add_edge("emit_query_answer_references", END)

related_qa_workflow = builder.compile()