import os
import re
import logging
import heapq
import json
import unicodedata

from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from typing import List, Literal, Dict, Any, Optional
import numpy as np
import textwrap

from llama_index.core.base.response.schema import Response
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever

from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.constants import Send
from rapidfuzz.fuzz import partial_ratio

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from registry import classify_and_subqueries_prompt

GROUNDED_PROMPT = PromptTemplate.from_template(
    """You are a helpful advisor and must respond in Norwegian (Bokmål).
    You MUST follow the rules below exactly.

    IMPORTANT PRINCIPLES:
    - You may ONLY use information from the 'context' (below). Do not add explanations, numbers, reasons, or advice that are not explicitly stated in the context.
    - Do not include any additional advice, opinions, or suggestions that are not explicitly present in the context.

    OUTPUT REQUIREMENTS:
    You MUST return valid JSON matching the Pydantic schema 'GroundedAnswer':
    {{
    "answer": str,
    "claims": List[str],
    "citations": [{{"doc_id": str, "quote": str}}]
    }}

    DEFINITIONS:
    - "answer":
    A coherent answer to the question, written in a friendly and clear manner, but ONLY based on the context.
    Do not include any information that is not directly found in the context.

    - "claims":
    A list of short, individual statements extracted from "answer".
    Each claim must be ONE clear sentence.
    Each claim MUST be directly supported by at least one quote in "citations".
    Do not combine multiple ideas from different parts of the context into one claim.
    Do not include anything that cannot be quoted verbatim from the context.

    - "citations":
    A list of evidence supporting the claims.
    For every "claim" in "claims", there must be at least one "citation".
    Each "citation" MUST include:
        - "doc_id": exactly the ID shown in brackets in the context, e.g. [https://ung.no/article] → "https://ung.no/article"
        - "quote": a VERBATIM text string (at least 8 characters long) copied directly from the context.

    VERY IMPORTANT:
        * Do not paraphrase the quote.
        * Do not add or remove words.
        * Do not merge two different parts using "..." to make it fit.
        * Do not change the word order.
        * Do not shorten a sentence into something that doesn’t exist as continuous text in the context.

    If you cannot find a verbatim "citation" in the context that supports a "claim", then that "claim" must NOT appear in "claims", and the information must NOT appear in the "answer" either.

    DO NOT:
    - Do not ask for more information.
    - Do not tell the user what to do, unless it is explicitly stated in the context.
    - Do not mention these instructions in your answer.

    QUESTION:
    {question}

    CONTEXT (SOURCES):
    Each source is labeled with an ID in brackets. Use it exactly as the doc_id.
    Only quote from these sources:

    {context}
    """
)




class Reference(TypedDict):
    name: str
    url: str
    relevancy_index: float   
    
# 1) Schema with required citations (doc_id + quote or span)
class Citation(BaseModel):
    doc_id: str
    quote: str = Field(..., min_length=8)

class GroundedAnswer(BaseModel):
    answer: str
    claims: List[str]
    citations: List[Citation]
    

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
    
    response_validity_index : float


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
    retriever: BaseRetriever
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
    
_POSSIBLE_META_IDS = ("doc_id", "from_doc_id", "document_id", "source_id", "file_id", "url")


def _collect_ids(node) -> list[str]:
    meta = getattr(node, "metadata", {}) or {}
    ids = [str(meta[k]) for k in _POSSIBLE_META_IDS if meta.get(k)]
    # include chunk id as a fallback
    chunk_id = getattr(node, "id_", None) or getattr(node, "node_id", None)
    if chunk_id:
        ids.append(str(chunk_id))
    return list(dict.fromkeys(ids))

def _preferred_display_id(node) -> str:
    ids = _collect_ids(node)
    return ids[0] if ids else "unknown"

def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()
    
def _node_id(n):
    # Prefer your own doc_id if present
    md = getattr(n, "metadata", {}) or {}
    return str(md.get("doc_id") or getattr(n, "id_", "unknown"))

def _node_text(n):
    # Works across TextNode/Document variants
    t = getattr(n, "text", None)
    if t:     
        return t #usually this one
    get_content = getattr(n, "get_content", None)
    if callable(get_content):     
        return get_content(metadata_mode="all") or ""
    return getattr(n, "get_text", lambda: "")() or ""

_WS = re.compile(r"\s+")
_TRANSLATE = str.maketrans({
    "\u2018": "'",  # ‘
    "\u2019": "'",  # ’
    "\u201C": '"',  # “
    "\u201D": '"',  # ”
    "\u2013": "-",  # –
    "\u2014": "-",  # —
    "\u00A0": " ",  # NBSP
    "\u202F": " ",  # NNBSP
})
_ZERO_WIDTH = dict.fromkeys(map(ord, ["\u200B", "\u200C", "\u200D", "\u2060"]), None)

def _normalize(s: str, *, collapse_ws: bool = True, case_sensitive: bool = False) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_TRANSLATE).translate(_ZERO_WIDTH)
    if collapse_ws:
        s = _WS.sub(" ", s)
    return s if case_sensitive else s.casefold()

def _format_context_from_nodes(nodes, max_chars_per_node=1800, max_nodes=6) -> str:
    parts = []
    for nws in nodes[:max_nodes]:
        node = getattr(nws, "node", nws)
        did = _preferred_display_id(node)   # <-- was _node_id(node)
        txt = _node_text(node).strip()
        if not txt:
            continue
        txt = txt[:max_chars_per_node]
        parts.append(f"[{did}]\n{textwrap.dedent(txt)}")
    return "\n\n".join(parts)

def _node_identity(n: Any) -> str:
    """Key for de-duplication while preserving order."""
    node = getattr(n, "node", n)
    return str(getattr(node, "id_", None) or getattr(node, "node_id", None) or id(node))


# --- main verifier: per-node matching, returns nodes with hits ---
# returns:
# problems: list of error strings (empty = all good)
# matched_nodes: de-duplicated list of nodes that matched at least one citation
# matches_by_citation: dict mapping citation index → list of nodes that matched that
#
def _verify_citations_per_node(
    citations: List["Citation"],
    nodes: List[Any],
    *,
    min_quote_chars: int = 8,
    collapse_whitespace: bool = True,
    case_sensitive: bool = False,
    # Optional fuzzy fallback: set to an int (e.g., 85) to enable
    fuzzy_min_ratio: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Check each citation.quote against each node's text (no combined corpus).

    Returns dict:
      {
        "problems": List[str],                       # empty == OK
        "matched_nodes": List[Any],                  # nodes that matched any citation (deduped, order-preserving)
        "matches_by_citation": Dict[int, List[Any]]  # citation index -> list of matching nodes
      }
    """
    problems: List[str] = []
    matches_by_citation: Dict[int, List[Any]] = {}
    matched_nodes_ordered: List[Any] = []
    seen_nodes: set[str] = set()

    # Pre-normalize node texts once
    norm_nodes = []
    for nws in nodes:
        text_raw = _node_text(nws)
        text_norm = _normalize(text_raw, collapse_ws=collapse_whitespace, case_sensitive=case_sensitive)
        norm_nodes.append((nws, text_norm))

    # If no text at all but we have citations
    if citations and not any(t for _, t in norm_nodes):
        return {
            "problems": [f"citation[{i}]: no retrieved text available" for i, _ in enumerate(citations)],
            "matched_nodes": [],
            "matches_by_citation": {},
        }

    # Check each citation against each node
    for i, cit in enumerate(citations):
        q_raw = (cit.quote or "").strip()
        q_len_norm = len(_normalize(q_raw, collapse_ws=True, case_sensitive=True))
        if q_len_norm < min_quote_chars:
            problems.append(f"citation[{i}]: quote too short (<{min_quote_chars})")
            continue

        q_norm = _normalize(q_raw, collapse_ws=collapse_whitespace, case_sensitive=case_sensitive)

        found_in_any = False
        for node_obj, node_text_norm in norm_nodes:
            meta = node_obj.metadata
            url = meta.get('url', 'Ingen URL')
            if not node_text_norm:
                continue

            hit = q_norm in node_text_norm
            if not hit and fuzzy_min_ratio is not None:
                try:
                    from rapidfuzz.fuzz import partial_ratio
                    hit = partial_ratio(q_norm, node_text_norm) >= fuzzy_min_ratio
                    print(f'------------->partial_ratio founf hit for the citation:{q_norm}<---------------------')
                except Exception:
                    hit = False

            if hit:
                found_in_any = True
                
                print(f'¤¤¤Fikk hit for {q_norm} for {url}')
                matches_by_citation.setdefault(i, []).append(node_obj)
                key = _node_identity(node_obj)
                if key not in seen_nodes:
                    seen_nodes.add(key)
                    matched_nodes_ordered.append(node_obj)

        if not found_in_any:
            problems.append(f"citation[{i}]: quote not found in any node: {q_raw!r}")
            print(f'¤¤¤¤Fikk IKKE hit for {q_norm}')

    return {
        "problems": problems,
        "matched_nodes": matched_nodes_ordered,
        "matches_by_citation": matches_by_citation,
    }
    
# ----------

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

_STOP = set("""
og i jeg det at en et den til er som på de med han av ikke for å var meg seg vi dere
oss din ditt deres dem da hun nå har om sin sitt sine også hadde hva skal selv mot dette
min mitt mine alle denne disse noen noe hver hvem vil bli ble blitt kunne må måtte innen
uten under over etter før mellom gjennom rundt fordi hvis mens når hvor hvorfor slik der
her hit dit opp ned ut inn igjen videre bare mye mange annen andre flere slik slik
fram tilbake alltid aldri kanskje allerede derfor enten eller både men så veldig ganske
hele helt noe nok samme noen gang ganger vår vårt våre hos blant innenfor utenfor ovenfor
nedenfor bak foran ved omkring utover innenfor innen utenfor uten gjennom imot tross selv
selv om deres dens deres hans hennes dens dens deres disse denne dette der den det de
dette disse vår vårt våre kunne skulle burde måtte være vært ble blir blitt gjør gjorde
gjort gjør gjørte gjortes hadde hatt har skal kan vil må bør får få fikk gi gitt sier sa
sagt ser så sånn slik at på i til av med om for fra mot gjennom blant ved mellom uten
oppe nede inne ute her der hit dit hvor når hvorfor hvem hvilken hvilket hvilke hvordan
hver alle alt annet andre noen noe ingen intet begge både hverken eller dersom fordi mens
når siden som hva hvor hvem hvilken hvilket hvilke hvorvidt hvorfor hvordan
""".split())

    
# Worker state
class WorkerState(TypedDict):
    subquery: SubQuery
    similarity_cutoff: float
    query_engine: BaseQueryEngine
    retriever: BaseRetriever
    llm: Any
    
def _emit(delta: str, event: str = "Answer"):
    writer = get_stream_writer()  
    writer({"event": event, "structured_answer_delta": delta})
    #logging.info({"event": event, "structured_answer_delta": delta})
    
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
        return {}
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
    state['subquery'].response_validity_index = best.score
    if (band != "Rejected"):
        state['subquery'].response_validity = "valid"
        state['subquery'].references = refs
        state['subquery'].answer = response_obj.response
    else: 
        state['subquery'].response_validity = "not valid"
        state['subquery'].answer = "Jeg beklager, men jeg kan bare svare på spørsmål basert på den gitte konteksten"

        
    return {}

def query_grounded(state: WorkerState) -> dict:
    writer = get_stream_writer()
    writer({"event": "info", "message": f"Worker answers the subquery \"{state['subquery'].subquery}\""})

    try:
        #qe = state["query_engine"]
        retriever = state["retriever"]
        question = state["subquery"].subquery

        # Retrieve
        #resp = qe.query(question)
        
        #nodes = getattr(resp, "source_nodes", []) or []
        nodes = retriever.retrieve(question) or []

        # Build grounded context and get structured output
        ctx = _format_context_from_nodes(nodes)
        
        chain = (
            RunnableLambda(lambda _: {"question": question, "context": ctx})
            | GROUNDED_PROMPT
            | state["llm"].with_structured_output(GroundedAnswer)
        )
        ga: GroundedAnswer = chain.invoke({})
        stats = _verify_citations_per_node(ga.citations, nodes, fuzzy_min_ratio=70)
        # Verify grounded citations (anywhere in retrieved text)
        print('\n-----------------------------------------------------------------------------------------------')
        print(f'*** Tester spørsmålet: <{question}>')
        print(f'*** Found context from invoke:{ctx}')
        problems = stats["problems"]
        matched_nodes = stats["matched_nodes"]
        print(f'*** ga.answer: <{ga.answer}>')
        #print(f'*** query.respons: <{resp.response}>')
        for i, cit in enumerate(ga.citations):
                q_raw = (cit.quote or "").strip()
                print(f'***** citation:{q_raw}')
        print(f'*** ga Problems: <{problems}>')
        print('\n-----------------------------------------------------------------------------------------------')

        
        refs: List[Reference] = []
        
        if not problems and ga.answer.strip() and "Det vet jeg ikke basert på kildene." not in ga.answer.strip():
            state["subquery"].response_validity = "valid"
            state["subquery"].answer = ga.answer  # ✅ prefer grounded
            
            
            for node in matched_nodes[:5]:
                if node.score is not None:
                    meta = node.metadata
                    refs.append({
                        "name": meta.get('title', 'Ingen tittel').lstrip(),
                        "url": meta.get('url', 'Ingen URL'),
                        "relevancy_index": node.score
                    })
            state["subquery"].references = refs
            print("====================================")
            print(state)
            print('====================================')
            
            return {}
        else:
            # Fallback: safe extractive summary (avoid hallucinations)
            top_text = "\n\n".join((_node_text(n) or "").strip()[:600] for n in nodes[:2] if _node_text(n))
            fallback = (
                "Jeg finner ikke støtte i kildene for alle detaljene. "
                #"Her er et utdrag fra kildene:\n\n" + top_text
            )
            state["subquery"].response_validity = "not valid"
            state["subquery"].answer = fallback
            
            print("====================================")
            print(state)
            print('====================================')
            return {}

    except Exception as e:
        logging.error(f"Failed to execute agent: {e}")
        state["subquery"].response_validity = "not valid"
        state["subquery"].answer = "Jeg klarte ikke å verifisere sitatene nå."
        return {}


def related_queries_and_categories(state: State_AnswerWithRelatedQueries) -> dict:
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
    combined =""
    combined_debug =""
    for s in sq:
        if s.response_validity == 'valid':
            combined = f"Subquery: {s.subquery}\n\n"
            combined += f"\nAnswer: {s.answer}\n\n"
            
            combined_debug = f"Subquery: {s.subquery}\n\n"
            combined_debug += f"\nAnswer: {s.answer}\n\n"
            combined_debug += f"\nRelevancy:{s.response_validity_index}"
            # if s.references:
            #     combined += "## Referanser\n"
            #     for r in s.references:
            #         combined += f"- [{r['name']}]({r['url']}) , relevans: {r['relevancy_index']:.2f}\n"
            completed_report_answers += combined  
        else:
            completed_report_answers_non_valid+=f'\n\nBeklager, men jeg kunne ikke svare på spørsmålet : \"{s.subquery}\"'
            combined_debug+=f'\n\nBeklager, men jeg kunne ikke svare på spørsmålet : \"{s.subquery}\"'
    print(f'+++++++++++++++++++++++++++++++++++++++++++')
    print(f's: <<{s}>>')
    print(f'combined_debug: <<{completed_report_answers}>>')
    print(f'+++++++++++++++++++++++++++++++++++++++++++')

    if completed_report_answers:
        aggregated_answer = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a friendly, empathetic, and knowledgeable health advisor designed to help young people in Norway (ages 13–19).\n\n"
                        "Your task is to synthesize a single, coherent final answer in Norwegian (Bokmål) **based only on the provided list of partial answers**.\n"
                        "You must **not invent, rephrase, or add new information, claims, explanations, or advice** that are not explicitly present in the provided text.\n"
                        "If something is missing, unclear, or contradictory, simply omit it — do not attempt to fill in or infer meaning.\n\n"

                        "**Your goal:**\n"
                        "- Combine overlapping or repeated points from the provided answers into a clean, readable summary.\n"
                        "- Preserve the factual content and tone exactly as written.\n"
                        "- Never introduce new claims, interpretations, or guidance.\n"
                        "- If all provided answers say 'Det vet jeg ikke basert på kildene.', then the final answer must **only** repeat that.\n\n"

                        "**Tone and Style Guidelines (must follow):**\n"
                        
                        "1. **Empathy:** The tone should be calm, kind, and supportive — but you may only express empathy if it already exists in the provided text. "
                        "Do not invent new emotional or motivational statements.\n\n"
                        
                        "2. **Teen-Friendly Language (Ages 13–19):**\n"
                        "- Keep language clear and natural, with short sentences.\n"
                        "- Avoid medical jargon unless it appears in the provided text.\n"
                        "- Do not expand or explain terms beyond what’s given.\n\n"
                        
                        "3. **Formatting and Structure:**\n"
                        "- Use concise headings expressing the query, only if they add value (e.g., \“Svar på spørsmål om <subquery>\””).\n"
                        "- Start directly with the synthesized answer — no preamble or meta text.\n"
                        "- Do **not** include the words 'Subquery', 'Sub-query', or similar.\n"
                        "- Do **not** add filler phrases like 'Here is…', 'Below is…', or 'Kort oppsummering'.\n"
                        "- Use simple markdown formatting and headings only if they already appear or clearly improve readability.\n\n"
                        "**IMPORTANT:**\n"
                        "You must never add tips, advice, or suggestions beyond what exists in the provided text.\n"
                        "If the provided answers contain no usable information, your entire output should simply be:\n"
                        "\"Det vet jeg ikke basert på kildene.\""
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
            "query_grounded",
            {
                "subquery": s,
                "query_engine": state["query_engine"],
                "similarity_cutoff": state["similarity_cutoff"],
                "llm": state["llm"],
                "retriever": state["retriever"]
            },
        )
        for s in state["subqueries"]
    ]

# Build workflow
builder = StateGraph(State_AnswerWithRelatedQueries)

# Add the nodes
builder.add_node("orchestrator", orchestrator)
builder.add_node("query_grounded", query_grounded)
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
    "orchestrator", assign_workers, ["query_grounded"]
)
builder.add_edge("query_grounded", "synthesizer")
builder.add_edge("synthesizer", "emit_query_answer_references")
builder.add_edge("emit_query_answer_references", "related_queries_and_categories")


builder.add_edge("related_queries_and_categories", END)

# Compile the workflow
answer_with_related_queries_workflow = builder.compile()

from graph_utils import save_mermaid_diagram
#save_mermaid_diagram(answer_with_related_queries_workflow.get_graph())



