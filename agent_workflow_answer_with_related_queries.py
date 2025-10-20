import os
import re
import logging
import heapq
import json
from typing import List, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from typing import List, Literal, Dict, Any, Optional

from llama_index.core.base.response.schema import Response
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever

from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.constants import Send

from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator

from registry import classify_and_subqueries_prompt

class Reference(TypedDict):
    name: str
    url: str
    relevancy_index: float   

# Schema for structured output to use in planning
class SubQuery(BaseModel):
    subquery: str = Field(
        description="The subquery",
    )
    answer: str = Field(
        description="Answer to the subquery",
    )
    references : List[Reference] = Field(
        description="list of references",
    )
    response_validity: Literal["valid", "not valid"]


class SubQueries(BaseModel):
    subqueries: List[SubQuery] = Field(
        description="Sections of the structured answer.",
    )
    main_category : str = Field (
        decription="The category for the user query"
    )
    query_severity : Literal["Green", "Yellow", "Red", ""] = Field (
        description = "the severity for the user query"
    )
    
class RelatedQuery(TypedDict):
    keyword: str
    query: str
# Graph state
   
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
    
    subqueries: list[SubQuery]  # List of subqueries
    final_answer: str  # Final report



def _get_field(o, k, default=None):
    return o.get(k, default) if isinstance(o, dict) else getattr(o, k, default)
    
# Worker state
class WorkerState(TypedDict):
    subquery: SubQuery
    similarity_cutoff: float
    query_engine: BaseQueryEngine
    
def _emit(delta: str, event: str = "Answer"):
    writer = get_stream_writer()  
    writer({"event": event, "structured_answer_delta": delta})
    logging.info({"event": event, "structured_answer_delta": delta})
    
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

def _mode_router(s: State_AnswerWithRelatedQueries) -> str:
    return "related_only" if s.get("related_only") else "full"


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

    return index_qa_bank.as_retriever(
        similarity_top_k=top_k,
        similarity_cutoff=cutoff,
        filters=composite,
    )
    
# Nodes
def orchestrator(state: State_AnswerWithRelatedQueries):
    """Orchestrator that generates a plan for solving the question"""
    writer = get_stream_writer()  
    writer({"event": "info", "message":"Orchestrator that generates a plan for solving the question"})
      
    # Generate queries
    llm = state["llm"]
        
    categories_json = json.dumps(state["categories"], ensure_ascii=False, indent=2)
    
    prompt = classify_and_subqueries_prompt(query=state["query"], categories=categories_json)
        
    # Augment the LLM with schema for structured output
    planner = llm.with_structured_output(SubQueries)

    report_queries = planner.invoke(
        prompt
    )
    
    cat_name = report_queries.main_category
    related_categories = _get_related_category_names(state["categories"], cat_name) if cat_name != "Ukjent" else []
    
    return {"subqueries": report_queries.subqueries, 
            "main_category": report_queries.main_category, 
            "query_severity": report_queries.query_severity,
            "related_categories": related_categories}

def llm_call(state: WorkerState):
    """Worker answers a subquery using the relevant index"""
    writer = get_stream_writer()  
    writer({"event": "info", "message" : f"Worker answers the subquery \"{state['subquery'].subquery}\""}) 
    
    query_engine = state["query_engine"]
    
    response_obj = query_engine.query(state['subquery'].subquery)
    
    cutoff = state["similarity_cutoff"]
 
    refs: List[Reference] = []

    
    thresholds = state.get("relevancy_thresholds", {
        "strong": 0.60,
        "medium": 0.55,   # anything below = Rejected
        "weak":   0.35,   # anything below = Rejected
    })



    nodes = [n for n in response_obj.source_nodes if n.score is not None]
    if not nodes:
        vector_index_desc = state["vector_index_description"]
        feedback = (f"Jeg beklager! {vector_index_desc}. "
                    f"Hvis du har spørsmål om disse emnene, kan jeg prøve å hjelpe.")
        return {

        }
    best = max(nodes, key=lambda n: n.score)
    
    band = _classify_relevancy(best.score, thresholds)
    
    logging.info(f"Band: {band} for {state['subquery'].subquery}, best score: {best}")
    
    for node in response_obj.source_nodes [:5]:
        if node.score is not None:
            meta = node.metadata
            refs.append({
                "name": meta.get('title', 'Ingen tittel').lstrip(),
                "url": meta.get('url', 'Ingen URL'),
                "relevancy_index": node.score
            })
    if (band != "Rejected"):
        state['subquery'].response_validity = "valid"
        state['subquery'].references = refs
        state['subquery'].answer = response_obj.response
    else: 
        state['subquery'].response_validity = "not valid"
        state['subquery'].answer = "Jeg beklager, men jeg kan bare svare på spørsmål basert på den gitte konteksten"
        
    return {}


def related_queries_and_categories(state: State_AnswerWithRelatedQueries) -> dict:
    """When related_only=True, make sure required fields exist without running answer/validation."""
    writer = get_stream_writer()
    

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
            text = r.node.get_text()
            sev = r.node.metadata.get("severity", "")
            cat = r.node.metadata.get("category", "")
            doc_id = r.node.metadata.get("from_doc_id", r.node.node_id)  # fallback if missing
            score = r.score
            #logging.info(f'candidate query: {text} | cat: {cat} | sev: {sev} | id: {doc_id}  | score: {score} ')
            candidates.append({"id": str(doc_id), "text": text, "severity": sev})
    
        related_queries = [{"keyword": s.get("severity", ""), "query": s["text"]} for s in candidates]

        related_queries_payload = json.dumps(related_queries, ensure_ascii=False)

        # Emit ONLY the JSON array (client listens for this event)
        _emit(related_queries_payload, event="Related queries")
                                           
        
    except Exception as e:
       logging.error(f"Failed to execute agent: {e} ")       
        
    return {"related_queries": related_queries, "related_categories": related_categories}
    
    
def synthesizer(state: State_AnswerWithRelatedQueries):
    """Synthesize full answer from answers from the subqueries"""
    
    writer = get_stream_writer()  
    writer({"event": "info", "message":"Synthesize full answer"}) 
    
    llm = state["llm"]
    
      # Get the list of SubQuery objects
    sq: list[SubQuery] = state["subqueries"]
    
    completed_report_answers=""
    completed_report_answers_non_valid=""
    
    for s in sq:
        if s.response_validity == 'valid':
            combined = f"Subquery: {s.subquery}\n\n"
            combined += f"\nAnswer: {s.answer}\n\n"
            # if s.references:
            #     combined += "## Referanser\n"
            #     for r in s.references:
            #         combined += f"- [{r['name']}]({r['url']}) , relevans: {r['relevancy_index']:.2f}\n"
            completed_report_answers += combined  
        else:
            completed_report_answers_non_valid+=f'\n\nBeklager, men jeg kunne ikke svare på spørsmålet : \"{s.subquery}\"'
            
    
    print(f'completed_report: <<{completed_report_answers}>>')

    if completed_report_answers:
        aggregated_answer = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "from this list of subqueries and answers, reorganize a final answer. Use markdown formatting.\n"
                        "STYLE RULES (must follow):\n"
                        "- Start directly with the answer. No preamble, no meta-text.\n"
                        "- Do NOT restate the question.\n"
                        "- Do NOT include the words “Subquery”, “Sub-query”, or any heading that begins with them.\n"
                        "- Do NOT include phrases like \“Here is…\”, \“Her er…\”, \“Below is…\”, \“Svar på spørsmålet ditt…\”, \"Kort oppsummering\".\n"
                        "- Use concise headings expressing the query, only if they add value (e.g., \“Svar på spørsmål om <subquery>\””).\n"
                        "- Output must be helpful, empathetic, youth-friendly, and in Norwegian Bokmål.\n"
                        "IF VIOLATED: Prefer omitting the section entirely rather than adding boilerplate.\n"
                    )
                ),
                HumanMessage( 
                    content=f"Here is the list of answers: {completed_report_answers}"
                ),
            ]
        )
  
        
        logging.info(f"Here is the list of answers: {completed_report_answers}")
        
        # build the most relavant references
        #
        ref_list =[]
        for s in sq:
            if s.references and s.response_validity == "valid":
                #combined += "\n## Referanser\n"
                for r in s.references:
                    ref_list.append(r)
                    
    ## keep best score per URL
        best_by_url = {}
        for r in ref_list:
            url = r.get("url") if isinstance(r, dict) else getattr(r, "url", None)
            score = r.get("relevancy_index") if isinstance(r, dict) else getattr(r, "relevancy_index", None)
            if url is None or score is None:
                continue  # skip malformed rows

            # keep the highest score per URL
            if (url not in best_by_url) or (score > best_by_url[url]["relevancy_index"]):
                best_by_url[url] = r

        ## take the 5 highest by relevancy_index
        top5 = heapq.nlargest(5, best_by_url.values(), key=lambda x: x["relevancy_index"])
        return {"final_answer": aggregated_answer.content + completed_report_answers_non_valid,
            "references": top5,
            }
    else:
        return {"final_answer": "Jeg beklager, men jeg kan bare svare på spørsmål basert på den gitte konteksten",
            "references": [],
            }
      
 

def emit_query_answer_references(state: State_AnswerWithRelatedQueries):
    
    
    
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

# Conditional edge function to create llm_call workers that each write a section of the report
def assign_workers(state: State_AnswerWithRelatedQueries):
    """Assign a worker to each section in the plan"""
    writer = get_stream_writer()  
    writer({"event": "info", "message":"Assign a worker to each section in the plan"}) 
  
    # Kick off section writing in parallel via Send() API
    return [
        Send(
            "llm_call",
            {
                "subquery": s,
                "query_engine": state["query_engine"],
                "similarity_cutoff": state["similarity_cutoff"],
            },
        )
        for s in state["subqueries"]
    ]

# Build workflow
builder = StateGraph(State_AnswerWithRelatedQueries)

# Add the nodes
builder.add_node("orchestrator", orchestrator)
builder.add_node("llm_call", llm_call)
builder.add_node("synthesizer", synthesizer)
builder.add_node("emit_query_answer_references", emit_query_answer_references)
builder.add_node("related_queries_and_categories", related_queries_and_categories)

# Add edges to connect nodes
builder.add_conditional_edges(
    START,
    _mode_router,
    {
        "related_only": "related_queries_and_categories",
        "full": "orchestrator",
    },
)
builder.add_conditional_edges(
    "orchestrator", assign_workers, ["llm_call"]
)
builder.add_edge("llm_call", "synthesizer")
builder.add_edge("synthesizer", "emit_query_answer_references")
builder.add_edge("emit_query_answer_references", "related_queries_and_categories")


builder.add_edge("related_queries_and_categories", END)

# Compile the workflow
answer_with_related_queries_workflow = builder.compile()

from graph_utils import save_mermaid_diagram
#save_mermaid_diagram(answer_with_related_queries_workflow.get_graph())



