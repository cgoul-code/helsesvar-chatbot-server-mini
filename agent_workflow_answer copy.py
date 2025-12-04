import os
import re
import logging
import heapq
import json
import unicodedata

from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from operator import add

from typing import List, Literal, Dict, Any, Optional, Annotated, Tuple
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

from langchain_core.runnables import RunnableLambda

from registry import classify_and_subqueries_prompt, GROUNDED_PROMPT





class Reference(TypedDict):
    name: str
    url: str
    relevancy_index: float   
    
# 1) Schema with required citations (doc_id + quote or span)

    
class Citation(BaseModel):
    url: str
    quote: str = Field(..., min_length=8)

class Claim(BaseModel):
    claim: str
    Citations: List[Citation]
    validity: Literal["valid", "not valid"]

class GroundedAnswer(BaseModel):
    answer: str
    claims: List[Claim]

    

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
    
    response_validity_index : float = 0.0


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
   
class State_Answer(TypedDict):
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
    from_node_id:str
    from_related_q: bool
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
    completed_subqueries: Annotated[list[SubQuery], add]
    final_answer: str  # Final report
    route: Literal["emit", "related_only", "full"]  
    
_POSSIBLE_META_IDS = ("doc_id", "from_doc_id", "document_id", "source_id")



def _wrap_at_nearest_space(text: str, width: int = 80) -> str:
    """
    Insert newline characters at the blank space nearest to every `width` chars.
    If no space exists near a cut, it hard-breaks at `width`.

    Existing newlines are respected (wrapped per line).
    """
    lines_out = []

    for original_line in text.splitlines():
        i = 0
        n = len(original_line)

        while i < n:
            # If the remainder fits, emit and stop
            if n - i <= width:
                lines_out.append(original_line[i:])
                break

            target = i + width

            # Search for the nearest *space* around the target
            prev_space = original_line.rfind(" ", i, min(n, target + 1))
            next_space = original_line.find(" ", target, n)

            if prev_space == -1 and next_space == -1:
                # No spaces at all in the rest: hard break
                lines_out.append(original_line[i:target])
                i = target
            else:
                # Choose the closest space to the target (prefer the left on a tie)
                if prev_space == -1:
                    cut = next_space
                elif next_space == -1:
                    cut = prev_space
                else:
                    cut = prev_space if (target - prev_space) <= (next_space - target) else next_space

                lines_out.append(original_line[i:cut])
                i = cut + 1  # skip the space we broke at

    return "\n".join(s.rstrip() for s in lines_out)

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
_CONTROL_CHARS = dict.fromkeys(range(0x00, 0x20), None)

def _normalize(s: str, *, collapse_ws: bool = True, case_sensitive: bool = False) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    # 1) Normaliser typografiske tegn
    s = s.translate(_TRANSLATE)
    # 2) Fjern zero-width
    s = s.translate(_ZERO_WIDTH)
    # 3) Fjern kontrolltegn (inkl. \x00)
    s = s.translate(_CONTROL_CHARS)
    # 4) Kollaps mellomrom
    if collapse_ws:
        s = _WS.sub(" ", s)
    return s if case_sensitive else s.casefold()

def _format_context_from_nodes(nodes, max_chars_per_node=3000, max_nodes=100) -> str:
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

    try: 
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
            print(f'--->original sitat:<{cit}>')
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
                print(f'1 {q_raw}')
                if not node_text_norm:
                    continue
                print('2')
                hit = q_norm in node_text_norm
                print(f'3: fuzzy_min_ratio {hit} {fuzzy_min_ratio}')
                if (not hit) and (fuzzy_min_ratio is not None):
                    print('3.1')
                    try:
                        ratio = partial_ratio(q_norm, node_text_norm )
                        hit = ratio >= fuzzy_min_ratio
                        print(f'------------->partial_ratio found hit for the citation:{q_norm}<---------------------')
                        print(f'hit: {hit}{ratio}')
                        print(f'------------->with text node: {node_text_norm}<------')
                    except Exception:
                        hit = False
                print('4')
                if hit:
                    found_in_any = True
                    
                    matches_by_citation.setdefault(i, []).append(node_obj)
                    key = _node_identity(node_obj)
                    if key not in seen_nodes:
                        seen_nodes.add(key)
                        matched_nodes_ordered.append(node_obj)
                        
                    break # go to next citation

            if not found_in_any:
                problems.append(f"citation[{i}]: quote not found in any node: {q_raw!r}")
                
    except Exception as e:
        logging.error(f"_verify_citations_per_node error: {e}")
        
    return {
        "problems": problems,
        "matched_nodes": matched_nodes_ordered,
        "matches_by_citation": matches_by_citation,
    }

    

from pydantic import BaseModel

def _verify_claims(
    grounded_answer: "GroundedAnswer",
    nodes: List[Any],
    *,
    min_quote_chars: int = 8,
    collapse_whitespace: bool = True,
    case_sensitive: bool = False,
    fuzzy_min_ratio: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Validate each claim in a GroundedAnswer.

    Returns a dict shaped like:
    {
        "global_problems": List[str],   # serious issues across all claims
        "claims_report": [
            {
                "claim_index": int,
                "claim_text": str,
                "validity_reported": str,          # "valid" | "not valid"
                "has_citations": bool,
                "all_citations_valid": bool,       # True if EVERY citation quote was actually found
                "any_citation_valid": bool,        # True if AT LEAST ONE citation was found
                "problems": List[str],             # problems for this claim
                "citations_report": [
                    {
                        "citation_index": int,
                        "url": str,
                        "quote": str,
                        "found_in_nodes": bool,
                        "problems": List[str],
                        "matched_node_urls": List[str],  # URLs of nodes where we saw the quote
                    },
                ],
            },
            ...
        ]
    }

    Notes:
    - We check that each claim has at least one citation.
    - We check that each citation actually appears in one of the provided nodes (using _verify_citations_per_node).
    - We cross-check the claim.validity field vs. what we observed.
    """

    global_problems: List[str] = []
    claims_report: List[Dict[str, Any]] = []

    # Pre-normalize node texts once (same logic as _verify_citations_per_node does internally)
    # We'll reuse _verify_citations_per_node per claim, but we also want quick access to URLs
    # for matched nodes for nicer reporting.
    # We let _verify_citations_per_node do the heavy lifting, since it's already careful
    # about normalization, fuzzy, etc.

    for claim_idx, claim_obj in enumerate(grounded_answer.claims):
        claim_text = claim_obj.claim
        validity_reported = claim_obj.validity
        claim_citations: List["Citation"] = claim_obj.Citations or []

        # Edge case: no citations at all
        if len(claim_citations) == 0:
            claim_problems = [
                f"claim[{claim_idx}]: has no citations"
            ]
            # if the model said validity == "valid", but gave no citations, that's a red flag
            if validity_reported == "valid":
                claim_problems.append(
                    f"claim[{claim_idx}]: validity='valid' but no citations provided"
                )

            claims_report.append({
                "claim_index": claim_idx,
                "claim_text": claim_text,
                "validity_reported": validity_reported,
                "has_citations": False,
                "all_citations_valid": False,
                "any_citation_valid": False,
                "problems": claim_problems,
                "citations_report": [],
            })
            global_problems.extend(claim_problems)
            continue

        # We have at least one citation. Validate them.
        cite_check = _verify_citations_per_node(
            claim_citations,
            nodes,
            min_quote_chars=min_quote_chars,
            collapse_whitespace=collapse_whitespace,
            case_sensitive=case_sensitive,
            fuzzy_min_ratio=fuzzy_min_ratio,
        )

        # cite_check["problems"]   -> e.g. ["citation[0]: quote too short", "citation[1]: quote not found..."]
        # cite_check["matches_by_citation"] -> {0:[nodeA], 2:[nodeC,...], ...}

        matches_by_citation = cite_check.get("matches_by_citation", {}) or {}
        per_claim_problems: List[str] = []

        # Build a per-citation report
        citations_report: List[Dict[str, Any]] = []
        any_citation_valid = False
        all_citations_valid = True  # we'll AND as we go

        for cit_i, cit in enumerate(claim_citations):
            # did this citation match?
            matched_nodes_for_cit = matches_by_citation.get(cit_i, [])
            found_in_nodes = len(matched_nodes_for_cit) > 0

            # Find any problems from cite_check that specifically reference this citation index
            this_cit_problems = [
                p for p in cite_check["problems"]
                if p.startswith(f"citation[{cit_i}]")
            ]

            # Update claim-level booleans
            if found_in_nodes:
                any_citation_valid = True
            else:
                all_citations_valid = False

            # Collect matched node URLs for debugging / audit
            matched_urls = []
            for n in matched_nodes_for_cit:
                try:
                    matched_urls.append(n.metadata.get("url", "Ingen URL"))
                except Exception:
                    matched_urls.append("Ingen URL")

            citations_report.append({
                "citation_index": cit_i,
                "url": cit.url,
                "quote": cit.quote,
                "found_in_nodes": found_in_nodes,
                "problems": this_cit_problems,
                "matched_node_urls": matched_urls,
            })

        # Merge all citation problems into this claim's problem list
        per_claim_problems.extend(cite_check["problems"])

        # Check logical consistency of 'validity'
        if validity_reported == "valid" and not any_citation_valid:
            per_claim_problems.append(
                f"claim[{claim_idx}]: validity='valid' but no citation actually matched any node"
            )
        if validity_reported == "not valid" and any_citation_valid:
            # this is optional: you may consider this a warning, not a hard problem
            per_claim_problems.append(
                f"claim[{claim_idx}]: validity='not valid' but at least one citation DID match"
            )

        claim_entry = {
            "claim_index": claim_idx,
            "claim_text": claim_text,
            "validity_reported": validity_reported,
            "has_citations": True,
            "all_citations_valid": all_citations_valid,
            "any_citation_valid": any_citation_valid,
            "problems": per_claim_problems,
            "citations_report": citations_report,
        }
        claims_report.append(claim_entry)

        # Add to global problems
        global_problems.extend(per_claim_problems)
        

    return {
        "global_problems": global_problems,
        "claims_report": claims_report,
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
    
def _mode_router(state: State_Answer) -> str:
    # value set by maybe_use_related_q
    r = state.get("route")
    if r in ("emit", "related_only", "full"):
        return r
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
    
def maybe_use_related_q(state: State_Answer):
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
    
# Nodes
def orchestrator(state: State_Answer):
    """Orchestrator that generates a plan for solving the question"""
    writer = get_stream_writer()  
    writer({"event": "info", "message":"Orchestrator that generates a plan for solving the question"})
    try:
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
        
    except Exception as e:
        logging.error(f"Failed to execute agent: {e}")
            
        # Trygg fallback så resten av grafen ikke krasjer
        return {
            "subqueries": [],
            "main_category": "Ukjent",
            "query_severity": "",
            "related_categories": [],
        }



def query_grounded(state: WorkerState) -> dict:
    writer = get_stream_writer()
    writer({"event": "info", "message": f"Worker answers the subquery \"{state['subquery'].subquery}\""})

    try:
        retriever = state["retriever"]
        question = state["subquery"].subquery

        # 1) Retrieve NodeWithScore objects
        nodes = retriever.retrieve(question) or []

        # 3) Validity check based on best score
        thresholds = state.get("relevancy_thresholds", {
            "strong": 0.60,
            "medium": 0.55,
            "weak":   0.35,
        })

        # Only consider nodes that actually have a numeric score
        if not nodes:
            # no scored nodes at all
            state["subquery"].response_validity = "not valid"
            state["subquery"].answer = ("Jeg beklager, men jeg kan bare svare på spørsmål basert på den gitte "
                                        "konteksten")
            return {"completed_subqueries": [state["subquery"]]}

        best_nws = max(nodes, key=lambda n: n.score)  # <- n is NodeWithScore
        best_score = float(getattr(best_nws, "score", 0.0))
        band = _classify_relevancy(best_score, thresholds)
        logging.info(f"Band: {band} for {question}, best score: {best_score:.3f}")

        # Build refs from the top few scored nodes
        refs: List[Reference] = []
        for nws in nodes[:5]:
            node_obj = getattr(nws, "node", nws)
            meta = getattr(node_obj, "metadata", {}) or {}
            refs.append({
                "name": (meta.get("title") or "Ingen tittel").lstrip(),
                "url": meta.get("url", "Ingen URL"),
                "relevancy_index": float(getattr(nws, "score", 0.0)),
            })

        state['subquery'].response_validity_index = best_score

        if band == "Rejected":
            state['subquery'].response_validity = "not valid"
            state['subquery'].answer = ("Jeg beklager, men jeg kan bare svare på spørsmål basert på den gitte "
                                        "konteksten")
            return {"completed_subqueries": [state["subquery"]]}
        
        # -----------------------------------------------------------------------------------------------
        # 4) Build grounded context from the ORIGINAL nodes (not the dicts)
        ctx = _format_context_from_nodes(nodes)

        chain = (
            RunnableLambda(lambda _: {"question": question, "context": ctx})
            | GROUNDED_PROMPT
            | state["llm"].with_structured_output(GroundedAnswer)
        )
        ga: GroundedAnswer = chain.invoke({})
        
        print(f'ga: {ga}')
        # 5) Validate claims against the ORIGINAL nodes
        #
        results = _verify_claims(
            ga,
            nodes,
            min_quote_chars=8,
            collapse_whitespace=True,
            case_sensitive=False,
            fuzzy_min_ratio=60,
        )

        # 6) Emit your systeminfo UI (unchanged)
        global_problems = results.get("global_problems", [])
        claims_report = results.get("claims_report", [])
        answer_wrapped = _wrap_at_nearest_space(ga.answer, width=120)

        # Intro
        _emit(f"## Delspørsmål: {question}", event="systeminfo")
        _emit(f"## Svar på delspørsmål:", event="systeminfo")
        _emit(f"{answer_wrapped}", event="systeminfo")
        _emit("\u00A0\n", event="systeminfo")
        _emit("\u00A0\n", event="systeminfo")
        _emit(" --- ", event="systeminfo")
        
        _emit("## Validering av påstander:", event="systeminfo")
        _emit("\n", event="systeminfo")

        # Iterate each claim
        state["subquery"].response_validity = "valid"
        for claim_entry in claims_report:
            idx = claim_entry["claim_index"]
            claim_text = claim_entry["claim_text"]
            validity_reported = claim_entry["validity_reported"]
            any_citation_valid = claim_entry["any_citation_valid"]
            all_citations_valid = claim_entry["all_citations_valid"]
            problems = claim_entry["problems"]
            citations_report = claim_entry["citations_report"]

            # Section header for this claim
            _emit(f"\n", event="systeminfo")
            _emit(f"# **Påstand {idx}: {claim_text}** ", event="systeminfo")
            #_emit("", event="systeminfo")

            # --- CLAIM SUMMARY:
            _emit(f"Minst én sitat-treff: {any_citation_valid}", event="systeminfo")
            _emit(f"Alle sitater gyldige: {all_citations_valid}", event="systeminfo")
      
            # Problems for this claim (if any)
            if problems:
                _emit("**Problemer for denne påstanden:**", event="systeminfo")
                for p in problems:
                    _emit(f"- {p}", event="systeminfo")
                _emit("", event="systeminfo")

            # --- CITATIONS TABLE (ASCII, monospaced) ---
            _emit("Sitat-tilknytning:", event="systeminfo")

            for cit in citations_report:
                cit_i = cit["citation_index"]
                found_in_nodes = cit["found_in_nodes"]
                urls = cit["matched_node_urls"]
                url_str = ""
                for i, u in enumerate(urls):
                    url_val = u or "Ingen URL"
                    url_str += f"[{url_val}]({url_val}) \n"
                    #print(f'url val:{url_val}')
                    
                quote_val = cit["quote"] or ""

                short_quote = quote_val.strip()
                if len(short_quote) > 140:
                    short_quote = short_quote[:137] + "..."
                short_quote = _wrap_at_nearest_space(short_quote, width=120)

                def esc(cell: str) -> str:
                    return cell.replace("|", "\\|")

                s = f"{cit_i} {'✅' if found_in_nodes else '❌'}  {short_quote} \n {esc(url_str)}"
                _emit(s, event="systeminfo")
            

            # Per-citation problems (if any)
            any_cit_problem = any(cit["problems"] for cit in citations_report)
            if any_cit_problem:
                _emit("**Detaljer per sitat:**", event="systeminfo")
                for cit in citations_report:
                    if not cit["problems"]:
                        continue
                    cit_i = cit["citation_index"]
                    _emit(f"- Sitat {cit_i}:", event="systeminfo")
                    for cp in cit["problems"]:
                        _emit(f"  - {cp}", event="systeminfo")
                _emit("", event="systeminfo")
                state["subquery"].response_validity = "not valid"

            # Divider between claims
            _emit("\u00A0\n", event="systeminfo")
            _emit("\u00A0\n", event="systeminfo")
            _emit(" --- ", event="systeminfo")
            _emit("\u00A0\n", event="systeminfo")

        res = state["subquery"].response_validity
        _emit(f"## Resultat: {res}", event="systeminfo")
        

        state["subquery"].answer = ga.answer
        state["subquery"].references = refs
        return {"completed_subqueries": [state["subquery"]]}

    except Exception as e:
        logging.error(f"Failed to execute agent: {e}")
        state["subquery"].response_validity = "not valid"
        state["subquery"].answer = "Jeg klarte ikke å verifisere sitatene nå."
        return {"completed_subqueries": [state["subquery"]]}


def related_queries_and_categories(state: State_Answer) -> dict:
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
    
    
def synthesizer(state: State_Answer):
    """Synthesize full answer from answers from the subqueries"""
    
    writer = get_stream_writer()  
    writer({"event": "info", "message":"Synthesize full answer"}) 

    
    llm = state["llm"]
    
    try:
        
        # Get the list of SubQuery objects
        sq: list[SubQuery] = state.get("completed_subqueries", []) or []
        
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

        if completed_report_answers:
            print(f"--->Agregerer disse svarene:{completed_report_answers}")
            aggregated_answer = llm.invoke(
                [
                    SystemMessage(
                        content = (
                            "Du er en vennlig, empatisk og kunnskapsrik helseveileder laget for å hjelpe ungdom i Norge (alder 13–19 år).\n\n"
                            "Din oppgave er å sette sammen et enkelt, helhetlig og sammenhengende svar på norsk (bokmål) **kun basert på den gitte listen med del-svar**.\n"
                            "Du skal **ikke finne på, omformulere eller legge til ny informasjon, påstander, forklaringer eller råd** som ikke uttrykkelig finnes i den gitte teksten.\n"
                            "Hvis noe mangler, er uklart eller motsier seg selv, skal du bare utelate det — ikke prøv å fylle inn eller tolke meningen.\n\n"

                            "**Målet ditt:**\n"
                            "- Slå sammen overlappende eller gjentatte poenger fra de gitte del-svarene til et ryddig og lettlest sammendrag.\n"
                            "- Behold det faktiske innholdet og tonen nøyaktig slik de er skrevet.\n"
                            "- Ikke legg til nye påstander, tolkninger eller veiledning.\n"
                            #"- Hvis alle del-svarene i \"completed_report_answers\" sier: «Det vet jeg ikke basert på kildene.», skal det endelige svaret **kun** gjenta det.\n\n"

                            "**Retningslinjer for tone og stil (må følges):**\n\n"
                            
                            "1. **Empati:** Tonen skal være rolig, vennlig og støttende — men du kan bare uttrykke empati dersom det allerede finnes i den gitte teksten. "
                            "Ikke finn på nye følelsesmessige eller motiverende utsagn.\n\n"
                            
                            "2. **Ungdomsvennlig språk (13-19 år):**\n"
                            "- Bruk et klart og naturlig språk med korte setninger.\n"
                            "- Unngå medisinske faguttrykk, med mindre de allerede finnes i teksten.\n"
                            "- Ikke utvid eller forklar begreper utover det som står skrevet.\n\n"
                            
                            "3. **Formatering og struktur:**\n"
                            "- Bruk korte overskrifter som uttrykker temaet for delspørsmålet.\n"
                            "- Start direkte med det sammensatte svaret — ikke bruk innledninger.\n"
                            "- Ikke henvise til konteksten, tekstene, referansetekstene, kildene eller artiklene\n"
                            "- Ikke legg til fyllord som 'Her er…', 'Nedenfor finner du…' eller 'Kort oppsummering'.\n"
                            "- Bruk enkel markdown-formatering og overskrifter bare hvis de allerede finnes, eller tydelig forbedrer lesbarheten.\n\n"
                            
                            "**VIKTIG:**\n"
                            "Du må aldri legge til tips, råd eller forslag utover det som finnes i teksten.\n"
                            "Hvis de gitte del-svarene ikke inneholder brukbar informasjon, skal hele svaret ditt rett og slett være:\n"
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
            
    except Exception as e:
       logging.error(f"Failed to execute agent: {e} ")        
 

def emit_query_answer_references(state: State_Answer):
    
    
    
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
def assign_workers(state: State_Answer):
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
builder = StateGraph(State_Answer)

# Add the nodes
builder.add_node("maybe_use_related_q", maybe_use_related_q)
builder.add_node("orchestrator", orchestrator)
builder.add_node("query_grounded", query_grounded)
builder.add_node("synthesizer", synthesizer, join=True)
builder.add_node("emit_query_answer_references", emit_query_answer_references)
builder.add_node("related_queries_and_categories", related_queries_and_categories)


# Correct start chain:
builder.add_edge(START, "maybe_use_related_q")
builder.add_conditional_edges(
    "maybe_use_related_q",
    _mode_router,
    {
        "emit": "emit_query_answer_references",
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
answer_workflow = builder.compile()

from graph_utils import save_mermaid_diagram
#save_mermaid_diagram(answer_with_related_queries_workflow.get_graph())



