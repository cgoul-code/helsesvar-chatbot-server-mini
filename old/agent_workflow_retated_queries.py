
import logging
import json
from typing import List, Literal, Dict, Any, Optional, Tuple
from typing_extensions import TypedDict
from pydantic import Field
from operator import add

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
   
class RelatedQuery(TypedDict):
    keyword: str
    query: str
# Graph state

class Reference(TypedDict):
    name: str
    url: str
    relevancy_index: float  
   
class State_Related(TypedDict):
    # infra
    llm: Any
    index: VectorStoreIndex                 # underlying text index (for _fetch_answer_from_related_question)
    index_related_queries: VectorStoreIndex # QA-bank index
    retriever_related_queries: BaseRetriever
    vector_index_description: str
    categories: List[Dict[str, Any]]

    # query context
    query: str
    main_category: str
    query_severity: Literal["Green", "Yellow", "Red", ""]

    # how we got here
    from_related_q: bool        # True if user clicked a related Q
    from_node_id: str           # node id of the QA-bank entry when from_related_q=True

    # similarity settings for related queries
    similarity_cutoff: float
    similarity_top_k: int

    # outputs
    related_categories: List[str]
    related_queries: List[RelatedQuery]
    references: List[Reference]
    final_answer: str
    route: Literal["emit", "related_only", "full"]  # but in this agent we only care about 'emit' and 'related_only'

def _mode_router(state: dict) -> str:
    r = state.get("route")
    if r in ("emit", "related_only", "full"):
        # In the related agent, treat "full" as "related_only"
        return "related_only" if r == "full" else r
    return "related_only" if state.get("related_only") else "full"

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
    ds = index.storage_context.docstore
    
    qa_node = ds_qa.get_node(node_id)
    

    # Best-scored node
    meta = getattr(qa_node, "metadata", {}) or {}
    answer = meta.get("answer", "")
    url = meta.get("url", "")
    # from_doc_id = meta.get("from_doc_id", "")
    
    # if from_doc_id:
    #     text_node = ds.get_node(from_doc_id)
    #     print(f'FOUND TEXTNODE')
    #     if text_node:
    #         text_meta = getattr(text_node, "metadata", {}) or {}
    #         title = text_meta.get("title", "")
    

    title = url
    
    category = meta.get("category", "")
    severity = meta.get("severity", "Green")



    if not answer :
        print('return no answer')
        return None

    # Build refs from top few
    refs: List[Reference] = []
    # for nws in nodes[:5]:
    #     n = getattr(nws, "node", nws)
    #     m = getattr(n, "metadata", {}) or {}
    
    ## TODO: qa-bank must store title for the url-doc the question is about
    
    refs.append({
        "name": url,
        "url": title,
        "relevancy_index": 0.0,
    })

    return answer, refs, category, severity
       
def _node_text(n):
    # Works across TextNode/Document variants
    t = getattr(n, "text", None)
    if t:     
        return t #usually this one
    get_content = getattr(n, "get_content", None)
    if callable(get_content):     
        return get_content(metadata_mode="all") or ""
    return getattr(n, "get_text", lambda: "")() or ""


def _emit(delta: str, event: str = "Answer"):
    writer = get_stream_writer()  
    writer({"event": event, "structured_answer_delta": delta})
    #logging.info({"event": event, "structured_answer_delta": delta})
    

def _build_category_index(categories: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {c["name"]: c for c in categories}

def _get_related(categories: List[Dict[str, Any]], name: str):
    idx = _build_category_index(categories)
    cat = idx.get(name)
    return cat.get("related", []) if cat else []

def _get_related_category_names(categories, name: str):
    return [r["name"] for r in _get_related(categories, name)]

def _as_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def _as_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

def _build_related_queries_retriever(index_qa_bank, *, top_k, cutoff, query_severity, main_category):
    
    top_k = _as_int(top_k, 5)                 # fallback sensible default
    cutoff = _as_float(cutoff, 0.0)
    
    # Allowed severities
    if query_severity == "Green":
        allowed_sev = ["Green"]
    elif query_severity == "Yellow":
        allowed_sev = ["Green", "Yellow"]
    else:
        allowed_sev = ["Green", "Yellow", "Red"]


    filters_list = [
        # # Include only nodes explicitly marked valid = True
        MetadataFilter(
            key="valid",
            value=1,
            operator=FilterOperator.EQ,
        ),
        MetadataFilter(
            key="severity",
            value=allowed_sev,
            operator=FilterOperator.IN,
        ),
    ]    
    

    # Optional category
    if main_category:
        if isinstance(main_category, str):
            filters_list.append(MetadataFilter(
                key="category", value=main_category, operator=FilterOperator.EQ
            ))
        else:
            filters_list.append(MetadataFilter(
                key="category", value=list(main_category), operator=FilterOperator.IN
            ))

    composite = MetadataFilters(filters=filters_list, condition="and")

    return index_qa_bank.as_retriever(
        similarity_top_k=top_k,
        similarity_cutoff=cutoff,
        filters=composite,
    )
    
def maybe_use_related_q(state: State_Related):
    """
    If from_related_q is True, try to answer directly from the Q→A index.
    On hit -> return final_answer/references and route='emit'.
    Otherwise route to related_only or full.
    """
    try:
        if state.get("from_related_q"):
            retr = state.get("retriever_related_queries")
            idx = state.get("index")
            idx_qa = state.get("index_related_queries")
            
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
                    "route": "emit",
                }

        # no fast hit; decide normal route
        return {
            "route": "related_only" if state.get("related_only") else "full"
        }

    except Exception as e:
        logging.error(f"maybe_use_related_q error: {e}")
        return {
            "route": "related_only" if state.get("related_only") else "full"
        }
        
def emit_query_answer_references(state: State_Related):
    
    writer = get_stream_writer()  
    writer({"event": "info", "message":"aggregating the final answer"}) 
    
    
    # emit query as a separate event
    q = state.get("query") 
    if q:
        _emit(json.dumps(q, ensure_ascii=False), event="Refined query")
        
    
    part = f"\n## Du spurte\n{state['query']}\n"
    _emit(part)

    part = f"\n## Svar\n"
    _emit(part)

    # Stream the answer line-by-line (or token-by-token if you have it):
    answer = state["final_answer"]
    for line in answer.splitlines(True):   # keep line breaks
        _emit(line)
    _emit("\n")
    
    
    # build the most relavant references
    #
    top5 = state["references"]
                
    
    if top5:
        _emit("\n## Referanser\n")
        for r in top5:
            bullet = f"- [{r['name']}]({r['url']}) \n"
            _emit(bullet)
    

    return {}

def related_queries_and_categories(state: State_Related) -> dict:
    """When related_only=True, make sure required fields exist without running answer/validation."""
    writer = get_stream_writer()
    related_queries = []
    related_categories = []
    

    try:
        # get the related categories
        #
        main_cat = state.get("main_category")
        logging.info(f"_get_related_category_names for main_cat:{main_cat}")
        related_categories = _get_related_category_names(state["categories"], main_cat) if main_cat != "Ukjent" else []

        _emit(json.dumps(main_cat, ensure_ascii=False), event="Maincategory")
        
        cats: list[str] = []
        seen = set(cats)
        for c in related_categories:
            if not c: continue
            s = str(c)
            if s not in seen:
                cats.append(s); seen.add(s)
        _emit(json.dumps(cats, ensure_ascii=False), event="Subcategories")  
        
        
        
        # assume you have index_qa_bank in state (or re-access via your vector_store)
        # uses main_category to retrieve queries relevant for refined_query
        retriever = _build_related_queries_retriever(
            index_qa_bank=state["index_related_queries"],            # ensure you stored this in state earlier
            top_k=state["similarity_top_k"],
            cutoff=state["similarity_cutoff"],
            query_severity=state.get("query_severity"),      # may be None → defaults to all
            main_category=state.get("main_category"),        # may be None → ignored
        )
        
        #print(f'---->query:', state["query"])

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
        _emit(related_queries_payload, event="Related queries")
                                           
        
    except Exception as e:
       logging.error(f"Failed to execute agent: {e} ")       
        
    return {"related_queries": related_queries, "related_categories": related_categories}
  
    

# Build workflow
related_builder = StateGraph(State_Related)

related_builder.add_node("maybe_use_related_q", maybe_use_related_q)
related_builder.add_node("related_queries_and_categories", related_queries_and_categories)
related_builder.add_node("emit_query_answer_references", emit_query_answer_references)

# Start with maybe_use_related_q
related_builder.add_edge(START, "maybe_use_related_q")

# Route based on maybe_use_related_q:
# - if from_related_q hit -> route='emit' -> emit_query_answer_references
# - else -> route='related_only' -> related_queries_and_categories
related_builder.add_conditional_edges(
    "maybe_use_related_q",
    _mode_router,   # you already have this
    {
        "emit": "emit_query_answer_references",
        "related_only": "related_queries_and_categories",
        # 'full' not used in this agent, we just treat it same as 'related_only'
    },
)

related_builder.add_edge("emit_query_answer_references", END)
related_builder.add_edge("related_queries_and_categories", END)

related_workflow = related_builder.compile()

from graph_utils import save_mermaid_diagram
#save_mermaid_diagram(answer_with_related_queries_workflow.get_graph())



