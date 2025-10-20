
import logging
import json
from typing import List, Literal, Dict, Any
from typing_extensions import TypedDict
from registry import qa_query_rerank_ids_prompt
from langgraph.config import get_stream_writer

from llama_index.core.base.response.schema import Response
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters,  FilterOperator
from llama_index.core import VectorStoreIndex

from langgraph.graph import StateGraph, START, END


# === Data types ===
class Reference(TypedDict):
    name: str
    url: str
    relevance_index: float

class RelatedQuery(TypedDict):
    keyword: str
    query: str

class State_AnswerWithRelatedQueries(TypedDict):
    related_only: bool
    llm: any  # LLM client (from server_settings.get_llm())
    index: VectorStoreIndex
    index_related_queries: VectorStoreIndex
    query_engine: BaseQueryEngine
    query_engine_related_queries: BaseQueryEngine
    retriever_related_queries: BaseRetriever
    vector_index_description: str
    main_category: str
    categories: List[Dict[str, Any]]
    query: str
    refined_query: str
    query_severity:Literal["Green", "Yellow", "Red", ""]
    related_categories: List[str]
    similarity_cutoff: float
    similarity_top_k: int
    relevancy_cutoff: float
    relevancy_band: str
    best_node_score: float
    response: Response | None
    validate_response_result: Literal["Accepted", "Rejected"]
    answer: str
    feedback: str
    references: List[Reference]
    related_queries: List[RelatedQuery]
    structured_answer: str


# === Helpers ===

def _build_category_index(categories: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {c["name"]: c for c in categories}

def _get_related(categories: List[Dict[str, Any]], name: str):
    idx = _build_category_index(categories)
    cat = idx.get(name)
    return cat.get("related", []) if cat else []

def _get_related_category_names(categories, name: str):
    return [r["name"] for r in _get_related(categories, name)]

def _classify_relevancy(score: float, thresholds: dict[str, float]) -> str:
    """
    thresholds: dict with descending levels, e.g.
        {"strong": 0.60, "medium": 0.45, "weak": 0.35}
    Returns one of: "Strong", "Medium", "Weak", "Rejected".
    """
    s = float(score)

    # Defaults if a key is missing
    strong = float(thresholds.get("strong", 0.60))
    medium = float(thresholds.get("medium", 0.50))
    weak   = float(thresholds.get("weak",   0.35))

    # Ensure ordering (desc). If someone passed bad values, sort them.
    # After this, strong >= medium >= weak.
    strong, medium, weak = sorted([strong, medium, weak], reverse=True)

    if s >= strong:
        return "Strong"
    if s >= medium:
        return "Medium"
    if s >= weak:
        return "Rejected"
    return "Rejected"


# === Node functions ===

def llm_refine_and_classify(state: State_AnswerWithRelatedQueries) -> dict:
    writer = get_stream_writer()
    writer({"event": "info", "message": "Renskriver og klassifiserer."})

    llm = state["llm"]
    categories_json = json.dumps(state["categories"], ensure_ascii=False, indent=2)

    from registry import refine_and_classify_prompt  # import the helper
    prompt = refine_and_classify_prompt(query=state["query"], categories=categories_json)

    msg = llm.invoke(prompt)
    try:
        obj = json.loads(msg.content)
    except Exception as e:
        # fail-soft: if parsing breaks, keep going with original query
        logging.error(f"refine_and_classify JSON parse error: {e} | raw={msg.content!r}")
        obj = {"refined_query": state["query"], "severity": "", "category": "Ukjent"}

    refined_query = obj.get("refined_query", state["query"])
    severity = obj.get("severity", "")
    cat_name = obj.get("category", "Ukjent")

    related = _get_related_category_names(state["categories"], cat_name) if cat_name != "Ukjent" else []

    return {
        "refined_query": refined_query,
        "query_severity": severity,
        "main_category": cat_name,
        "related_categories": related,
    }



def llm_call_answer(state: State_AnswerWithRelatedQueries) -> dict:
    writer = get_stream_writer()
    writer({"event": "info", "message": "Svarer."})
    
    response_obj = state["query_engine"].query(state["refined_query"])
    #logging.info(f'---step---: llm_call_answer:{response_obj.response}')
    return {"answer": response_obj.response, "response": response_obj}



def allowed_severities(qsev: str | None) -> set[str]:
    if qsev == "Green":
        return {"Green"}
    if qsev == "Yellow":
        return {"Green", "Yellow"}
    return {"Green", "Yellow", "Red"}  # default / Red

from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters, FilterOperator

def build_related_queries_retriever(index_qa_bank, *, top_k, cutoff, query_severity, main_category):
    # Allowed severities
    if query_severity == "Green":
        allowed_sev = ["Green"]
    elif query_severity == "Yellow":
        allowed_sev = ["Green", "Yellow"]
    else:
        allowed_sev = ["Green", "Yellow", "Red"]

    filters_list = [
        # one flat filter using IN — no nested MetadataFilters
        MetadataFilter(key="severity", value=allowed_sev, operator=FilterOperator.IN)
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
    
    #print("composite:", composite)

    return index_qa_bank.as_retriever(
        similarity_top_k=top_k,
        similarity_cutoff=cutoff,
        filters=composite,
    )

def ensure_related_only_defaults(state: State_AnswerWithRelatedQueries) -> dict:
    """When related_only=True, make sure required fields exist without running answer/validation."""
    writer = get_stream_writer()
    def emit(delta: str, event: str = "chunk"):
        writer({"event": event, "structured_answer_delta": delta})
    try:
        # assume you have index_qa_bank in state (or re-access via your vector_store)
        # uses main_category to retrieve queries relevant for refined_query
        retriever = build_related_queries_retriever(
            index_qa_bank=state["index_related_queries"],            # ensure you stored this in state earlier
            top_k=state["similarity_top_k"],
            cutoff=state["similarity_cutoff"],
            query_severity=state.get("query_severity"),      # may be None → defaults to all
            main_category=state.get("main_category"),        # may be None → ignored
        )

        results = retriever.retrieve(state["refined_query"])

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
            text = r.node.get_text()
            sev = r.node.metadata.get("severity", "")
            cat = r.node.metadata.get("category", "")
            doc_id = r.node.metadata.get("from_doc_id", r.node.node_id)  # fallback if missing
            score = r.score
            #logging.info(f'candidate query: {text} | cat: {cat} | sev: {sev} | id: {doc_id}  | score: {score} ')
            candidates.append({"id": str(doc_id), "text": text, "severity": sev})
    
        related_queries = [{"keyword": s.get("severity", ""), "query": s["text"]} for s in candidates]
        
        # get the related categories
        #
        cat_name = state["main_category"]
        related_categories = _get_related_category_names(state["categories"], cat_name) if cat_name != "Ukjent" else []
        
        
    except Exception as e:
       logging.error(f"Failed to execute agent: {e} ")       
        
    return {"related_queries": related_queries, "related_categories": related_categories}
    
def llm_call_related_queries(state: State_AnswerWithRelatedQueries) -> dict:

    # Step 1: Retrieve candidates from vector index
    writer = get_stream_writer()
    writer({"event": "info", "message": "Henter lignende spørsmål."})
    
    try:


        # assume you have index_qa_bank in state (or re-access via your vector_store)
        retriever = build_related_queries_retriever(
            index_qa_bank=state["index_related_queries"],            # ensure you stored this in state earlier
            top_k=state["similarity_top_k"],
            cutoff=state["similarity_cutoff"],
            query_severity=state.get("query_severity"),      # may be None → defaults to all
            main_category=state.get("main_category"),        # may be None → ignored
        )

        results = retriever.retrieve(state["refined_query"])



        if results:
            def score_or_min(r):
                return float("-inf") if (getattr(r, "score", None) is None) else r.score

            max_idx = max(range(len(results)), key=lambda i: score_or_min(results[i]))
            max_score = score_or_min(results[max_idx])

            if max_score > 0.7:
                #logging.info(f"Dropping top candidate with score {max_score:.3f} (> 0.7)")
                results = [r for i, r in enumerate(results) if i != max_idx]        

        candidates = []
    
        for r in results[:5]:
            text = r.node.get_text()
            sev = r.node.metadata.get("severity", "")
            doc_id = r.node.metadata.get("from_doc_id", r.node.node_id)  # fallback if missing
            score = r.score
            #logging.info(f'candidate query: {text} | sev: {sev} | id: {doc_id}  | score: {score} ')
            candidates.append({"id": str(doc_id), "text": text, "severity": sev})
    

        # Step 2: Build prompt
        # candidates_jsonl = "\n".join(json.dumps(c, ensure_ascii=False) for c in candidates)
        # prompt = qa_query_rerank_ids_prompt(
        #     max_results=5,
        #     user_query=state["refined_query"],
        #     candidates_jsonl=candidates_jsonl,
        # )
        # logging.info(f'PROMPT:\n{prompt}')

        # # Step 3: Call LLM (through query_engine)
        # raw_resp = state["query_engine"].query(prompt).response
        # logging.info(f'LLM raw response: {raw_resp}')

        # try:
        #     selected_ids = json.loads(raw_resp).get("selected_ids", [])
        # except Exception as e:
        #     logging.error(f"Failed to parse LLM response: {e} | raw={raw_resp!r}")
        #     selected_ids = []

        # Step 4: Map back to original queries (unchanged)
        # selected = [c for c in candidates if c["id"] in selected_ids]
        
        # selected_id_set = set(selected_ids)
        # not_selected = [c for c in candidates if c["id"] not in selected_id_set]
        
        # for n_s in not_selected:
        #     print("n_s", n_s)
        
        related = [{"keyword": s.get("severity", ""), "query": s["text"]} for s in candidates]
        
    except Exception as e:
       logging.error(f"Failed to execute agent: {e} ")
        
    return {"related_queries": related}
    
 
    

def validate_response(state: State_AnswerWithRelatedQueries) -> dict:
    writer = get_stream_writer()
    writer({"event": "info", "message": "Validerer."})
    
    # Helper to emit a chunk the client can append
    # (Use a consistent shape so your client knows how to handle it.)
    def emit(delta: str, event: str = "chunk"):
        writer({"event": event, "structured_answer_delta": delta})

    thresholds = state.get("relevancy_thresholds", {
        "strong": 0.60,
        "medium": 0.50,   # anything below = Rejected
        "weak":   0.35,   # anything below = Rejected
    })

    nodes = [n for n in state["response"].source_nodes if n.score is not None]
    if not nodes:
        vector_index_desc = state["vector_index_description"]
        feedback = (f"Jeg beklager! {vector_index_desc}. "
                    f"Hvis du har spørsmål om disse emnene, kan jeg prøve å hjelpe.")
        return {
            "validate_response_result": "Rejected",
            "feedback": feedback,
            "relevancy_band": "Rejected",
            "best_node_score": None,
        }

    best = max(nodes, key=lambda n: n.score)
    #for n in nodes:
        #print(f'score:{n.score}')
        
    print(f'best score:{best.score}')
 

    band = _classify_relevancy(best.score, thresholds)

    # Keep your current two-way routing, but annotate the band + best score.
    if band == "Rejected":
        vector_index_desc = state["vector_index_description"]
        feedback = (f"Jeg beklager! "
                    f"Hvis du har spørsmål om disse emnene: \"{vector_index_desc}\", kan jeg prøve å hjelpe.")
        return {
            "validate_response_result": "Rejected",
            "feedback": feedback,
            "relevancy_band": band,
            "best_node_score": best.score,
        }
    else:
        combined_parts = []
        part = f"## Du spurte\n{state['refined_query']}\n"
        combined_parts.append(part); emit(part)

        part = f"## Svar\n"
        combined_parts.append(part); emit(part)

        # Stream the answer line-by-line (or token-by-token if you have it):
        answer = state.get("answer") or ""
        for line in answer.splitlines(True):   # keep line breaks
            combined_parts.append(line)
            emit(line)
        emit("\n")
        
        return {
            "validate_response_result": "Accepted",
            "relevancy_band": band,
            "best_node_score": best.score,
        }


def on_reject_build_structured(state: State_AnswerWithRelatedQueries) -> dict:
    # exactly what aggregator does:
    return aggregator(state)

def response_builder_node(state: State_AnswerWithRelatedQueries) -> dict:
    return {}


def references_generator(state: State_AnswerWithRelatedQueries) -> dict:
    writer = get_stream_writer()
    writer({"event": "info", "message": "Henter referansene."})
    
    relevancy_cutoff = state["relevancy_cutoff"]
    refs: List[Reference] = []
    seen = set()  # track already added (e.g. by URL)

    for node in state["response"].source_nodes:
        if node.score is not None and node.score >= relevancy_cutoff:
            meta = node.metadata
            name = meta.get('title', 'Ingen tittel').lstrip()
            url = meta.get('url', 'Ingen URL')
            key = (name, url)  # use tuple to identify uniqueness

            if key not in seen:
                seen.add(key)
                refs.append({
                    "name": name,
                    "url": url,
                    "relevance_index": node.score
                })
    return {"references": refs}

def aggregator(state: State_AnswerWithRelatedQueries) -> dict:
    writer = get_stream_writer()
    writer({"event": "info", "message": "Ferdigstiller."})

    def emit(delta: str, event: str = "chunk"):
        #print({"event": event, "structured_answer_delta": delta})
        writer({"event": event, "structured_answer_delta": delta})
        
    rfq = state.get("refined_query") 
    if rfq:
         emit(json.dumps(rfq, ensure_ascii=False), event="Refined query")
        
 
        
    rq = (state.get("related_queries") or [])[:5]
    emit(json.dumps(rq, ensure_ascii=False), event="Related queries")

    main_cat = state.get("main_category")
    emit(json.dumps(main_cat, ensure_ascii=False), event="Maincategory")
    
    rel = state.get("related_categories") or []
    cats: list[str] = []
    seen = set(cats)
    for c in rel:
        if not c: continue
        s = str(c)
        if s not in seen:
            cats.append(s); seen.add(s)
    emit(json.dumps(cats, ensure_ascii=False), event="Subcategories")
    
    if state.get("related_only"):
        writer({"event": "done"})
        return {"structured_answer": ""}

    # If rejected: stream the feedback and finish
    if state["validate_response_result"] == "Rejected":
        feedback = state["feedback"] or "Beklager, jeg fant ikke relevant info."
        emit(feedback + "\n")
        writer({"event": "done"})
        return {"structured_answer": feedback}

    combined_parts = []

    # References (stream each bullet)
    refs = state.get("references") or []
    if refs:
        header = "## Referanser\n"
        combined_parts.append(header); emit(header)
        for r in refs:
            bullet = f"- [{r['name']}]({r['url']}) \n"
            combined_parts.append(bullet); emit(bullet)

    # Related queries
    rq = (state.get("related_queries") or [])[:5]
    # if rq:
    #     header = "## Lignende spørsmål\n"
    #     combined_parts.append(header); emit(header)
    #     for r in rq:
    #         q = r.get("query") or r.get("text") or r.get("q")
    #         if q:
    #             line = f"=={q}==\n"
    #             combined_parts.append(line); emit(line)
    related_queries_payload = json.dumps(rq, ensure_ascii=False)

    # Emit ONLY the JSON array (client listens for this event)
    emit(related_queries_payload, event="Related queries")

    
    # emit system info
    best_score = state["best_node_score"]
    writer({"event": "systeminfo", "message": f"best score:{best_score: .2f}%"})
    
    # Finish stream
    writer({"event": "done"})

    combined = "".join(combined_parts)
    
    
    return {"structured_answer": combined}



# === Build static, stateless workflow ===
builder = StateGraph(State_AnswerWithRelatedQueries)

# 1️⃣ Core answer + validation
builder.add_node("llm_call_answer", llm_call_answer)
builder.add_node("validate_response", validate_response)
builder.add_node("llm_refine_and_classify", llm_refine_and_classify)
builder.add_node("aggregator", aggregator)
builder.add_node("references_generator", references_generator)
builder.add_node("llm_call_related_queries", llm_call_related_queries)
builder.add_node("ensure_related_only_defaults", ensure_related_only_defaults) 

def _mode_router(s: State_AnswerWithRelatedQueries) -> str:
    return "related_only" if s.get("related_only") else "full"

builder.add_conditional_edges(
    START,
    _mode_router,
    {
        "related_only": "ensure_related_only_defaults",
        "full": "llm_refine_and_classify",
    },
)

# Branch A: related_only fast path
builder.add_edge("ensure_related_only_defaults", "llm_call_related_queries")
builder.add_edge("llm_call_related_queries", "aggregator")

# Branch B: full flow
builder.add_edge("llm_refine_and_classify", "llm_call_answer")
builder.add_edge("llm_call_answer", "validate_response")


# 2️⃣ Branch on validation result:
#    - "Rejected" → aggregator
#    - "Accepted" → fan-out into 4 nodes



builder.add_conditional_edges(
    "validate_response",
    # router: return exactly the keys below
    lambda s: "Rejected" if s["validate_response_result"] == "Rejected"
              else "Accepted",
    {
        "Rejected": "aggregator",
        "Accepted": "references_generator",
    }
)

# 3️⃣ Fan-out the Accepted branch into the other 3 nodes
#    (these edges will only fire when validate_response_result == "Accepted")

#builder.add_edge("validate_response", "references_generator")


builder.add_edge("references_generator", "llm_call_related_queries")
builder.add_edge("llm_call_related_queries", "aggregator")


# 5️⃣ Finally, aggregator → END
builder.add_edge("aggregator", END)

answer_with_related_queries_workflow = builder.compile()

# produce graph.mmd that visualizes the workflow
from graph_utils import save_mermaid_diagram
#save_mermaid_diagram(answer_with_related_queries_workflow.get_graph())

logging.info("answer_witth_related_queries_workflow created...")