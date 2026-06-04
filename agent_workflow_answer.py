import os
import json
import logging
import re
import textwrap
import unicodedata
import heapq
import random


from operator import add
from typing import Any, Dict, List, Literal, Optional, Tuple, Annotated


from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator

from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer
from langgraph.constants import Send

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.callbacks import UsageMetadataCallbackHandler

from rapidfuzz.fuzz import partial_ratio

from registry import (
    subqueries_prompt,
    GROUNDED_PROMPT,
    CANNOT_ANSWER_PLACEHOLDER,
    ANALYZE_QUERY_PROMPT,
    STYLE_WARM_PROMPT,
    STYLE_SUPPORTIVE_PROMPT,
    STYLE_CRISIS_PROMPT,
    HARM_REFUSAL_ANSWER,
    HARM_REFUSAL_SHORT_ANSWER,
    HELP_AFTER_HARM_ANSWER,
    HELP_AFTER_HARM_SHORT_ANSWER,
    REFUSE_HARM_PROMPT,
    HELP_AFTER_HARM_PROMPT,
)

# Gyldige response-style-verdier. 'factual' = skip rewrite. Hvis klienten
# sender en ukjent verdi (eller ingen), faller vi tilbake til auto-routing
# via pick_response_style(severity, stance).
RESPONSE_STYLES = ("factual", "warm", "supportive", "crisis")

# Mapping fra stil-id til prompt-template. 'factual' har ingen prompt –
# apply_response_style hopper over LLM-callen og returnerer svaret uendret.
_STYLE_TO_PROMPT = {
    "warm": STYLE_WARM_PROMPT,
    "supportive": STYLE_SUPPORTIVE_PROMPT,
    "crisis": STYLE_CRISIS_PROMPT,
}

from agent_shared import Reference, _emit, _node_text, _build_related_queries_retriever, _as_int, _as_float, _dedupe_references, _normalize

import typing
import typing_extensions
typing.TypedDict = typing_extensions.TypedDict

# Hvor mange noder sender vi inn i LLM-konteksten?
MAX_NODES_FOR_CONTEXT = 8

# Hvor mange noder bruker vi til å verifisere sitater?
MAX_NODES_FOR_VERIFICATION = 8

# Hvor mange tegn per node som brukes
MAX_CHARS_PER_NODE = 2500

# ---------------------------------------------------------
# Datamodeller og typer
# ---------------------------------------------------------

class Citation(BaseModel):
    url: str
    quote: str = Field(..., min_length=8)


class Claim(BaseModel):
    claim: str
    Citations: List[Citation]
    validity: Literal["valid", "not valid"]


class GroundedAnswer(BaseModel):
    answer: str
    short_answer: str
    claims: List[Claim]


class SubQuery(BaseModel):
    subquery: str = Field(description="The subquery")
    answer: str = Field(description="Answer to the subquery")
    short_answer: str = Field(default="", description="Short summarized version for the answer")
    references: List[Reference] = Field(description="List of references")
    response_validity: Literal["valid", "not valid"]
    response_validity_index: float = 0.0


class SubQueries(BaseModel):
    subqueries: List[SubQuery] = Field(
        description="Sections of the structured answer."
    )
    main_category: str = Field(
        description="The category for the user query"
    )
    query_severity: Literal["Green", "Yellow", "Red", ""] = Field(
        description="The severity for the user query"
    )


class RelatedQuery(TypedDict):
    keyword: str
    query: str
    # node_id er lagt til i bruk, men ikke i TypedDict – legg til her hvis du vil type-sikre det:
    # node_id: str
    
class RelatedSelection(BaseModel):
    selected_node_ids: List[str] = Field(default_factory=list, max_items=3)
    rationale: str = Field(default="", description="Kort begrunnelse (valgfritt)")


class State_Answer(TypedDict):
    ''' config params'''
    llm: Any
    
    index: VectorStoreIndex
    query_engine: BaseQueryEngine # Text-bank
    retriever: BaseRetriever
    
    index_related_queries: VectorStoreIndex # QA-bank index
    retriever_related_queries: BaseRetriever

    vector_index_description: str
    query: str
    conversation_str: str
    from_node_id: str
    similarity_cutoff: float
    similarity_top_k: int
    relevancy_cutoff: float

    ''' router / plan '''
    needs_subqueries: bool  # <--- NY
    
    ''' calculated params'''
    refined_query: str
    main_category: str
    query_severity: Literal["Green", "Yellow", "Red", ""]
    stance: Literal["info_seeker", "affected_party", "harm_to_others", "ambiguous"]
    harm_to_others_tense: Literal["planning", "completed", "unclear", "na"]
    # Response-style. Tom streng = auto-rute via pick_response_style(). En
    # av RESPONSE_STYLES = klient-override som hopper over auto-routingen.
    response_style: str
    # Hvordan response_style ble valgt: 'auto' (auto-routet), 'override'
    # (klient sendte en gyldig stil), 'forced_red' (klamret til crisis pga
    # Red-severity safety floor), eller '' (apply_response_style ble
    # hoppet over – f.eks. harm-routing eller tomt svar).
    response_style_source: Literal["auto", "override", "forced_red", ""]
    relevancy_band: str
    best_node_score: float
    validate_response_result: Literal["Accepted", "Rejected"]
    answer: str
    feedback: str
    references: List[Reference]
    subqueries: List[SubQuery]
    completed_subqueries: Annotated[List[SubQuery], add]
    final_answer: str
    final_short_answer: str
    
    input_tokens: Annotated[int, add]
    output_tokens: Annotated[int, add]

    ''' tuning '''
    # Min fraction of cited claims that must be supported for an answer to
    # stay "valid" in query_grounded. Default 1.0.
    claims_valid_threshold: float
    # When True, query_grounded runs an LLM entailment gate that downgrades
    # claims whose (real) quote doesn't support them. Default True.
    entailment_check: bool

    ''' debug '''
    debug_emit_nodes: bool  # emit exact retrieved nodes as `retrieved_nodes`

class NextIntent(BaseModel):
    intent: str = Field(description="Kort intensjon, ikke et fullstendig spørsmål")
    why: str
    importance: float = Field(ge=0.0, le=1.0)

class NextIntents(BaseModel):
    intents: List[NextIntent] = Field(min_items=1, max_items=4)

class CandidateScore(BaseModel):
    node_id: str
    score: float = Field(ge=0.0, le=1.0)
    rationale: str

class RerankResult(BaseModel):
    ranked: List[CandidateScore]
    
class DialogPlan(BaseModel):
    last_user_question: str = Field(description="Siste brukerspørsmål")
    intents: List[NextIntent] = Field(min_items=1, max_items=4)

class RefusalResponse(BaseModel):
    """Strukturert output for refuse_harm_to_others og help_after_harm."""
    answer: str = Field(description="Hele svaret i markdown.")
    short_answer: str = Field(description="Én setning som oppsummerer svaret.")


class QueryPlan(BaseModel):
    refined_query: str = Field(
        description="Rewritten, clear version of user's question in Norwegian."
    )
    needs_subqueries: bool = Field(
        description=(
            "True if the question should be decomposed into multiple subqueries "
            "to answer it properly, False if a single answer is enough."
        )
    )
    query_severity: Literal["Green", "Yellow", "Red", ""] = Field(
        description=(
            "Define the severity (ALVORLIGHESGRAD) for the query")
    )
    stance: Literal["info_seeker", "affected_party", "harm_to_others", "ambiguous"] = Field(
        default="info_seeker",
        description=(
            "Brukerens rolle i situasjonen: 'info_seeker' (ber om generell info), "
            "'affected_party' (beskriver noe som rammer brukeren selv), "
            "'harm_to_others' (ber om hjelp til å påføre andre skade — straffbart "
            "eller åpenbart skadelig), eller 'ambiguous' (uklart hvem som er aktør)."
        ),
    )
    harm_to_others_tense: Literal["planning", "completed", "unclear", "na"] = Field(
        default="na",
        description=(
            "Bare relevant når stance='harm_to_others'. Tempus for handlingen: "
            "'planning' (brukeren vurderer/planlegger), 'completed' (brukeren "
            "har allerede gjort det), 'unclear' (tvetydig). Sett 'na' når "
            "stance ikke er 'harm_to_others'."
        ),
    )

class WorkerState(TypedDict):
    subquery: SubQuery
    similarity_cutoff: float
    query_engine: BaseQueryEngine
    retriever: BaseRetriever
    llm: Any
    conversation_str: str
    query_severity: str
    claims_valid_threshold: float
    entailment_check: bool
    debug_emit_nodes: bool
    

_POSSIBLE_META_IDS = ("doc_id", "from_doc_id", "document_id", "source_id")


# ---------------------------------------------------------
# Små hjelpefunksjoner
# ---------------------------------------------------------

def _pick_cannot_answer_placeholder(query_severity: str) -> str:
    sev = (query_severity or "").strip()  # "Green" | "Yellow" | "Red" | ""
    if not sev:
        sev = "Green"  # sensible default

    eligible = [
        x for x in CANNOT_ANSWER_PLACEHOLDER
        if sev in (x.get("severity") or [])
    ]

    # Fallback if list is misconfigured or severity missing
    if not eligible:
        eligible = CANNOT_ANSWER_PLACEHOLDER

    return random.choice(eligible)["answer"]

# Generisk wrapper:
def _invoke_with_usage(llm, messages) -> tuple[Any, int, int]:
    """
    Kaller llm.invoke med UsageMetadataCallbackHandler.
    messages kan være str, PromptValue, eller List[BaseMessage].
    Returnerer (result, input_tokens, output_tokens).
    """
    callback = UsageMetadataCallbackHandler()
    result = llm.invoke(messages, config={"callbacks": [callback]})
    in_tok, out_tok = _extract_usage_tokens(callback.usage_metadata)
    return result, in_tok, out_tok

def _make_dialog_plan(llm, history_txt: str) -> DialogPlan:
    prompt = (
        "Du får en samtalehistorikk i én tekststreng. Den inneholder flere tidligere spørsmål og svar.\n\n"
        "Oppgaver:\n"
        "1) Finn og skriv ut siste BRUKER-spørsmål (kort, uten ekstra tekst).\n"
        "2) Foreslå 2-4 sannsynlige NESTE intensjoner som en naturlig fortsettelse i dialogen.\n"
        "   Intensjoner skal være korte beskrivelser, ikke fullstendige spørsmål.\n"
        "   Unngå å gjenta siste brukerspørsmål.\n\n"
        f"Samtalehistorikk:\n{history_txt}\n"
    )

    return llm.with_structured_output(DialogPlan).invoke(prompt)

def _extract_usage_tokens(usage_meta: dict) -> tuple[int, int]:
    """Summerer input/output tokens fra UsageMetadataCallbackHandler.usage_metadata."""
    in_tokens = 0
    out_tokens = 0
    if not usage_meta:
        return 0, 0

    for _model_name, data in usage_meta.items():
        # Strukturen er typisk:
        # { "gpt-4.1-mini": {"input_tokens": 123, "output_tokens": 45, ...}, ... }
        in_tokens += int(data.get("input_tokens", 0) or 0)
        out_tokens += int(data.get("output_tokens", 0) or 0)

    return in_tokens, out_tokens

def _wrap_at_nearest_space(text: str, width: int = 80) -> str:
    """
    Bryt linjer ved nærmeste blanktegn rundt `width` tegn.
    Respekter eksisterende linjeskift.
    """
    lines_out: List[str] = []

    for original_line in text.splitlines():
        i = 0
        n = len(original_line)

        while i < n:
            if n - i <= width:
                lines_out.append(original_line[i:])
                break

            target = i + width
            prev_space = original_line.rfind(" ", i, min(n, target + 1))
            next_space = original_line.find(" ", target, n)

            if prev_space == -1 and next_space == -1:
                lines_out.append(original_line[i:target])
                i = target
            else:
                if prev_space == -1:
                    cut = next_space
                elif next_space == -1:
                    cut = prev_space
                else:
                    cut = prev_space if (target - prev_space) <= (next_space - target) else next_space

                lines_out.append(original_line[i:cut])
                i = cut + 1  # hopp over mellomrommet

    return "\n".join(s.rstrip() for s in lines_out)


def _collect_ids(node) -> List[str]:
    meta = getattr(node, "metadata", {}) or {}
    ids = [str(meta[k]) for k in _POSSIBLE_META_IDS if meta.get(k)]
    chunk_id = getattr(node, "id_", None) or getattr(node, "node_id", None)
    if chunk_id:
        ids.append(str(chunk_id))
    # bevar rekkefølge og fjern duplikater
    return list(dict.fromkeys(ids))


def _preferred_display_id(node) -> str:
    ids = _collect_ids(node)
    return ids[0] if ids else "unknown"


def _format_context_from_nodes(
    nodes: List[Any],
    max_chars_per_node: int = MAX_CHARS_PER_NODE,
    max_nodes: int = 100,
) -> str:
    """Formaterer noder til en tekstlig kontekst til LLM."""
    parts: List[str] = []
    for nws in nodes[:max_nodes]:
        node = getattr(nws, "node", nws)
        did = _preferred_display_id(node)
        txt = _node_text(node).strip()
        if not txt:
            continue
        truncated = txt[:max_chars_per_node]
        
        # finn siste avsnitt eller punktum
        last_break = max(
            truncated.rfind("\n\n"),
            truncated.rfind(". "),
        )
        txt =  truncated[:last_break + 1] if last_break > max_chars_per_node // 2 else truncated
        
        parts.append(f"[{did}]\n{textwrap.dedent(txt)}")
    return "\n\n".join(parts)


def _node_identity(n: Any) -> str:
    """Key for de-duplication while preserving order."""
    node = getattr(n, "node", n)
    return str(
        getattr(node, "id_", None) or
        getattr(node, "node_id", None) or
        id(node)
    )


def _node_meta(n: Any) -> Dict[str, Any]:
    node = getattr(n, "node", n)
    return getattr(node, "metadata", {}) or {}


def _ensure_article_in_top(nodes: List[Any], top_n: int) -> List[Any]:
    """Sørg for at minst én artikkel-node er med i topp-N hvis en kvalifiserer.

    I unified-indeksen ('Hyb' mode) konkurrerer article- og qa-noder i samme
    retrieval. Hvis topp-N er ren qa, men minst én artikkel finnes lenger
    ned i lista (dvs. har klart similarity_cutoff), byttes den lavest-
    rangerte qa-noden i topp-N ut med den høyest-rangerte artikkelen.
    Hvis det ikke finnes noen qa å bytte ut, legges artikkelen til.
    """
    if not nodes:
        return list(nodes)

    top = list(nodes[:top_n])
    if any(_node_meta(n).get("node_type") == "article" for n in top):
        return top

    best_article = next(
        (n for n in nodes if _node_meta(n).get("node_type") == "article"),
        None,
    )
    if best_article is None:
        return top

    qa_positions = [
        i for i, n in enumerate(top)
        if _node_meta(n).get("node_type") == "qa"
    ]
    if qa_positions:
        top[qa_positions[-1]] = best_article
    else:
        top.append(best_article)
    return top


# ---------------------------------------------------------
# Validering av sitater og påstander
# ---------------------------------------------------------

def _verify_citations_per_node(
    citations: List[Citation],
    nodes: List[Any],
    *,
    min_quote_chars: int = 8,
    collapse_whitespace: bool = True,
    case_sensitive: bool = False,
    fuzzy_min_ratio: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Sjekk hver citation.quote mot hver node (per-node matching).

    Returnerer:
      {
        "problems": List[str],
        "matched_nodes": List[Any],
        "matches_by_citation": Dict[int, List[Any]]
      }
    """
    problems: List[str] = []
    matches_by_citation: Dict[int, List[Any]] = {}
    matched_nodes_ordered: List[Any] = []
    seen_nodes: set[str] = set()

    try:
        # Pre-normaliser nodetekster
        norm_nodes: List[Tuple[Any, str]] = []
        for nws in nodes:
            text_raw = _node_text(nws)
            text_norm = _normalize(
                text_raw,
                collapse_ws=collapse_whitespace,
                case_sensitive=case_sensitive,
            )
            norm_nodes.append((nws, text_norm))

        if citations and not any(t for _, t in norm_nodes):
            return {
                "problems": [
                    f"citation[{i}]: no retrieved text available"
                    for i, _ in enumerate(citations)
                ],
                "matched_nodes": [],
                "matches_by_citation": {},
            }

        for i, cit in enumerate(citations):
            q_raw = (cit.quote or "").strip()
            q_len_norm = len(
                _normalize(q_raw, collapse_ws=True, case_sensitive=True)
            )
            if q_len_norm < min_quote_chars:
                problems.append(
                    f"citation[{i}]: quote too short (<{min_quote_chars})"
                )
                continue

            q_norm = _normalize(
                q_raw,
                collapse_ws=collapse_whitespace,
                case_sensitive=case_sensitive,
            )

            found_in_any = False
            for node_obj, node_text_norm in norm_nodes:
                if not node_text_norm:
                    continue

                hit = q_norm in node_text_norm

                if (not hit) and (fuzzy_min_ratio is not None):
                    try:
                        ratio = partial_ratio(q_norm, node_text_norm)
                        hit = ratio >= fuzzy_min_ratio
                        if hit:
                            logging.debug(
                                "Fuzzy match for citation %r, ratio=%s",
                                q_norm,
                                ratio,
                            )
                    except Exception:
                        hit = False

                if hit:
                    found_in_any = True
                    matches_by_citation.setdefault(i, []).append(node_obj)
                    key = _node_identity(node_obj)
                    if key not in seen_nodes:
                        seen_nodes.add(key)
                        matched_nodes_ordered.append(node_obj)
                    # gå videre til neste citation
                    break

            if not found_in_any:
                problems.append(
                    f"citation[{i}]: quote not found in any node: {q_raw!r}"
                )

    except Exception as e:
        logging.error("_verify_citations_per_node error: %s", e)

    return {
        "problems": problems,
        "matched_nodes": matched_nodes_ordered,
        "matches_by_citation": matches_by_citation,
    }


def _verify_claims(
    grounded_answer: GroundedAnswer,
    nodes: List[Any],
    *,
    min_quote_chars: int = 8,
    collapse_whitespace: bool = True,
    case_sensitive: bool = False,
    fuzzy_min_ratio: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Valider hver claim i et GroundedAnswer mot nodene.

    Returnerer:
    {
        "global_problems": List[str],
        "claims_report": [
            {
                "claim_index": int,
                "claim_text": str,
                "validity_reported": str,
                "has_citations": bool,
                "all_citations_valid": bool,
                "any_citation_valid": bool,
                "problems": List[str],
                "citations_report": [
                    {
                        "citation_index": int,
                        "url": str,
                        "quote": str,
                        "found_in_nodes": bool,
                        "problems": List[str],
                        "matched_node_urls": List[str],
                    },
                ],
            },
            ...
        ]
    }
    """
    global_problems: List[str] = []
    claims_report: List[Dict[str, Any]] = []

    for claim_idx, claim_obj in enumerate(grounded_answer.claims):
        claim_text = claim_obj.claim
        validity_reported = claim_obj.validity
        claim_citations: List[Citation] = claim_obj.Citations or []

        # Ingen sitater
        if len(claim_citations) == 0:
            claim_problems = [f"claim[{claim_idx}]: has no citations"]

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

        # Vi har minst ett sitat – verifiser
        cite_check = _verify_citations_per_node(
            claim_citations,
            nodes,
            min_quote_chars=min_quote_chars,
            collapse_whitespace=collapse_whitespace,
            case_sensitive=case_sensitive,
            fuzzy_min_ratio=fuzzy_min_ratio,
        )

        matches_by_citation = cite_check.get("matches_by_citation", {}) or {}
        per_claim_problems: List[str] = []
        citations_report: List[Dict[str, Any]] = []

        any_citation_valid = False
        all_citations_valid = True

        for cit_i, cit in enumerate(claim_citations):
            matched_nodes_for_cit = matches_by_citation.get(cit_i, [])
            found_in_nodes = len(matched_nodes_for_cit) > 0

            this_cit_problems = [
                p for p in cite_check["problems"]
                if p.startswith(f"citation[{cit_i}]")
            ]

            if found_in_nodes:
                any_citation_valid = True
            else:
                all_citations_valid = False

            matched_urls: List[str] = []
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

        per_claim_problems.extend(cite_check["problems"])

        if validity_reported == "valid" and not any_citation_valid:
            per_claim_problems.append(
                f"claim[{claim_idx}]: validity='valid' but no citation actually matched any node"
            )
        if validity_reported == "not valid" and any_citation_valid:
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
        global_problems.extend(per_claim_problems)

    return {
        "global_problems": global_problems,
        "claims_report": claims_report,
    }


# ---------------------------------------------------------
# Kategorier og relaterte spørsmål
# ---------------------------------------------------------

def _classify_relevancy(score: float, thresholds: Dict[str, float]) -> str:
    """
    thresholds: f.eks {"strong": 0.60, "medium": 0.45, "weak": 0.35}
    Returnerer: "Strong", "Medium" eller "Rejected".
    """
    s = float(score)

    strong = float(thresholds.get("strong", 0.60))
    medium = float(thresholds.get("medium", 0.50))
    weak = float(thresholds.get("weak", 0.35))

    strong, medium, weak = sorted([strong, medium, weak], reverse=True)

    if s >= strong:
        return "Strong"
    if s >= medium:
        return "Medium"
    if s >= weak:
        return "Rejected"
    return "Rejected"




# ---------------------------------------------------------
# Noder i grafen
# ---------------------------------------------------------
def analyze_query(state: State_Answer) -> Dict[str, Any]:
    """Renskriver spørsmålet og avgjør om vi trenger subqueries."""

    _emit("Analyze and possibly rewrite user query", event="info")

    llm = state["llm"]

    conversation_str = state.get("conversation_str", "")
    original_q = state["query"]

    prompt = ANALYZE_QUERY_PROMPT.format(
        conversation_str=conversation_str,
        original_q=original_q,
    )

    planner = llm.with_structured_output(QueryPlan)
    plan: QueryPlan = planner.invoke(prompt)


    # Vi skriver om query i state til renskrevet versjon
    print(f'Severity: {plan.query_severity}, Stance: {plan.stance}, Needs subqueries: {plan.needs_subqueries}, Harm tense: {plan.harm_to_others_tense}')
    return {
        "refined_query": plan.refined_query,
        "needs_subqueries": plan.needs_subqueries,
        "query_severity": plan.query_severity,
        "stance": plan.stance,
        "harm_to_others_tense": plan.harm_to_others_tense,
    }
    
def orchestrator(state: State_Answer) -> Dict[str, Any]:
    """Planlegger – genererer delspørsmål basert på brukerens spørsmål."""
    
    _emit("Orchestrator that generates a plan for solving the question", event="info")

    try:
        llm = state["llm"]
        prompt = subqueries_prompt(query=state["refined_query"])

        planner = llm.with_structured_output(SubQueries)
        report_queries: SubQueries = planner.invoke(prompt)

        return {
            "subqueries": report_queries.subqueries,
            "main_category": report_queries.main_category or "",
        }

    except Exception as e:
        logging.error("Failed to execute orchestrator: %s", e)
        return {
            "subqueries": []
        }

def fast_single(state: State_Answer) -> Dict[str, Any]:
    """
    Fasttrack: bruk query_grounded på ett enkelt spørsmål,
    og sett final_answer + references direkte.
    Respekterer response_validity som er satt basert på claims_report.
    """

    _emit("Fasttrack: answer single refined question without subqueries", event="info")

    # Lag en SubQuery med det renskrevne spørsmålet
    subq = SubQuery(
        subquery=state["refined_query"],
        answer="",
        short_answer="",
        references=[],
        response_validity="not valid",
        response_validity_index=0.0,
    )

    worker_state: WorkerState = {
        "subquery": subq,
        "similarity_cutoff": state["similarity_cutoff"],
        "query_engine": state["query_engine"],
        "retriever": state["retriever"],
        "llm": state["llm"],
        "conversation_str": state.get("conversation_str", ""),
        "query_severity": state.get("query_severity", "Green"),
        "claims_valid_threshold": state.get("claims_valid_threshold", 1.0),
        "entailment_check": state.get("entailment_check", True),
        "debug_emit_nodes": state.get("debug_emit_nodes", False),
    }

    # Kjør eksisterende logikk (henter noder, genererer GroundedAnswer,
    # kjører _verify_claims, setter response_validity osv.)
    result = query_grounded(worker_state)

    completed_list = result.get("completed_subqueries") or [subq]
    completed = completed_list[0]

    in_tokens = result.get("input_tokens", 0) or 0
    out_tokens = result.get("output_tokens", 0) or 0
    
    placeholder = _pick_cannot_answer_placeholder(state.get("query_severity", "Green"))
    final_answer = placeholder
    final_short_answer = placeholder
    
    refs = []
    
    # Determine validate_response_result based on validity
    validate_response_result = "Rejected"  # default

    # ---------- HER BRUKER VI CLAIMS-VALIDERINGA ----------
    if completed.response_validity == "valid":
        final_answer = completed.answer or final_answer
        final_short_answer = completed.short_answer or ""
        refs = _dedupe_references(completed.references or [], top_k=5)
        validate_response_result = "Accepted"  # override if valid
    # ------------------------------------------------------

    # Oppdater state-feltene som brukes senere
    return {
        "completed_subqueries": [completed],
        "final_answer": final_answer,
        "final_short_answer": final_short_answer,
        "references": refs,
        "input_tokens": (state.get("input_tokens", 0) or 0) + in_tokens,
        "output_tokens": (state.get("output_tokens", 0) or 0) + out_tokens,
        "validate_response_result": validate_response_result,
    }
    
def route_after_analysis(state: State_Answer) -> str:
    """Bestem neste node basert på stance, tense og needs_subqueries.

    Stance dominerer. For 'harm_to_others' deler vi i to grener basert på
    tense: 'completed' går til help_after_harm (skadebegrensning, ikke
    avvisning), alt annet går til refuse_harm_to_others. Dette unngår at
    kunnskapsbasen (som er skrevet for ofre/generell info) blir brukt til
    å besvare et gjerningsperson-spørsmål.
    """
    if state.get("stance") == "harm_to_others":
        tense = state.get("harm_to_others_tense", "unclear")
        if tense == "completed":
            return "help_after_harm"
        return "refuse_harm_to_others"
    if state.get("needs_subqueries"):
        return "orchestrator"
    return "fast_single"


def refuse_harm_to_others(state: State_Answer) -> Dict[str, Any]:
    """Konstruktivt avslag når brukeren PLANLEGGER å skade andre.

    LLM-drevet for å håndtere ulike harm-kategorier generisk (deling av
    bilder, overvåking av partner, trusler, catfishing, hevn-ideasjon
    osv.). Faller tilbake til hardkodet melding hvis LLM-callen feiler.
    """
    _emit("Stance=harm_to_others (planning/unclear): refusal node", event="info")

    llm = state["llm"]
    query = state.get("refined_query") or state.get("query") or ""
    tense = state.get("harm_to_others_tense", "unclear") or "unclear"
    conversation_str = state.get("conversation_str", "") or ""

    try:
        prompt_value = REFUSE_HARM_PROMPT.format(
            query=query,
            tense=tense,
            conversation_str=conversation_str,
        )
        result, in_tokens, out_tokens = _invoke_with_usage(
            llm.with_structured_output(RefusalResponse),
            prompt_value,
        )
        return {
            "final_answer": result.answer,
            "final_short_answer": result.short_answer,
            "references": [],
            "validate_response_result": "Accepted",
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
        }
    except Exception as e:
        logging.error("refuse_harm_to_others LLM call failed, using static fallback: %s", e)
        return {
            "final_answer": HARM_REFUSAL_ANSWER,
            "final_short_answer": HARM_REFUSAL_SHORT_ANSWER,
            "references": [],
            "validate_response_result": "Accepted",
            "input_tokens": 0,
            "output_tokens": 0,
        }


def help_after_harm(state: State_Answer) -> Dict[str, Any]:
    """Konstruktiv hjelp når brukeren HAR utført noe som har skadet noen.

    Ikke en avvisning – fokus på skadebegrensning, ta ansvar, kontakt
    voksen/politi/advokat, hjelp den rammede, og få hjelp selv. LLM-
    drevet, faller tilbake til hardkodet melding ved feil.
    """
    _emit("Stance=harm_to_others (completed): help_after_harm node", event="info")

    llm = state["llm"]
    query = state.get("refined_query") or state.get("query") or ""
    conversation_str = state.get("conversation_str", "") or ""

    try:
        prompt_value = HELP_AFTER_HARM_PROMPT.format(
            query=query,
            conversation_str=conversation_str,
        )
        result, in_tokens, out_tokens = _invoke_with_usage(
            llm.with_structured_output(RefusalResponse),
            prompt_value,
        )
        return {
            "final_answer": result.answer,
            "final_short_answer": result.short_answer,
            "references": [],
            "validate_response_result": "Accepted",
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
        }
    except Exception as e:
        logging.error("help_after_harm LLM call failed, using static fallback: %s", e)
        return {
            "final_answer": HELP_AFTER_HARM_ANSWER,
            "final_short_answer": HELP_AFTER_HARM_SHORT_ANSWER,
            "references": [],
            "validate_response_result": "Accepted",
            "input_tokens": 0,
            "output_tokens": 0,
        }


# ---------------------------------------------------------
# Entailment-gate: et sitat kan finnes ordrett i kilden uten å støtte
# påstanden (f.eks. et p-stav-sitat brukt på en p-plaster-påstand). Sitat-
# sjekken (_verify_claims) fanger bare oppdiktede sitater, ikke gale
# slutninger — denne porten lukker det hullet.
# ---------------------------------------------------------

class _EntailmentItem(BaseModel):
    index: int
    supported: bool = Field(
        description="True bare hvis sitatet/sitatene faktisk støtter påstanden."
    )

class _EntailmentResult(BaseModel):
    verdicts: List[_EntailmentItem]


ENTAILMENT_PROMPT = (
    "Du er en streng faktasjekker. For hver PÅSTAND får du ett eller flere SITAT "
    "som er hentet ordrett fra kildene. Avgjør om sitatet/sitatene FAKTISK STØTTER "
    "påstanden — altså om de handler om SAMME tiltak/forhold og logisk medfører "
    "påstanden.\n"
    "- Et sitat som handler om noe ANNET (for eksempelet annet produkt/prevensjonsmiddel, en "
    "annen aldersgruppe, en annen ordning) støtter IKKE påstanden, selv om noen ord "
    "er like.\n"
    "- Vær spesielt streng på forveksling av ulike, men beslektede ting "
    "(f.eks. p-stav vs p-plaster vs p-pille).\n"
    "Returner for hver indeks om den er supported (true/false).\n\n"
    "Påstander og sitater:\n"
)


def _content_tokens(text: str) -> set:
    """Distinctive (len>=5) lowercase tokens used for the cheap pre-filter."""
    return {
        t for t in re.findall(r"[a-zæøå0-9]+", (text or "").lower())
        if len(t) >= 5
    }


def _entailment_needed(claim: str, quotes: List[str], min_overlap: float = 0.9) -> bool:
    """True if claim/quote term-overlap is low enough to warrant an LLM check.

    Cheap gate so obviously-supported claims skip the LLM entirely. With high
    overlap we assume support; otherwise we let the LLM decide.
    """
    ct = _content_tokens(claim)
    if not ct:
        return False
    qt: set = set()
    for q in quotes:
        qt |= _content_tokens(q)
    present = sum(1 for t in ct if t in qt)
    return (present / len(ct)) < min_overlap


def _apply_entailment_gate(claims_report: List[Dict[str, Any]], llm) -> Tuple[int, int]:
    """Downgrade string-matched claims whose quotes don't actually support them.

    Runs ONE batched LLM call over the claims a cheap term-overlap pre-filter
    flags as suspicious. Fails open: any error keeps the existing string-match
    validity. Returns (input_tokens, output_tokens) consumed.
    """
    candidates: List[Tuple[Dict[str, Any], List[str]]] = []
    for ce in claims_report:
        if not ce.get("any_citation_valid"):
            continue
        quotes = [
            (c.get("quote") or "").strip()
            for c in ce.get("citations_report", [])
            if c.get("found_in_nodes") and (c.get("quote") or "").strip()
        ]
        if not quotes:
            continue
        if not _entailment_needed(ce.get("claim_text", ""), quotes):
            continue
        candidates.append((ce, quotes))

    if not candidates:
        return 0, 0

    lines = []
    for i, (ce, quotes) in enumerate(candidates):
        joined = " | ".join(quotes)
        lines.append(f"{i}. PÅSTAND: {ce.get('claim_text', '')}\n   SITAT: {joined}")
    prompt = ENTAILMENT_PROMPT + "\n".join(lines)

    try:
        res, in_tok, out_tok = _invoke_with_usage(
            llm.with_structured_output(_EntailmentResult), prompt
        )
        supported = {v.index: v.supported for v in res.verdicts}
    except Exception as e:
        logging.error("Entailment batch call failed, keeping string-match validity: %s", e)
        return 0, 0

    for i, (ce, _quotes) in enumerate(candidates):
        # Default True = don't downgrade a claim the model didn't return a verdict for.
        if supported.get(i, True):
            continue
        ce["any_citation_valid"] = False
        ce["all_citations_valid"] = False
        note = (
            f"claim[{ce.get('claim_index')}]: sitatet finnes i kilden, men støtter "
            f"ikke påstanden (entailment-sjekk)"
        )
        ce.setdefault("problems", []).append(note)
        cits = ce.get("citations_report") or []
        if cits:
            cits[0].setdefault("problems", []).append(note)
        _emit(
            f"⛔ Entailment: påstand {ce.get('claim_index', 0) + 1} forkastet — "
            f"sitatet handler om noe annet enn påstanden.",
            event="systeminfo",
        )

    return in_tok, out_tok


def query_grounded(state: WorkerState) -> Dict[str, Any]:
    """
    For hvert delspørsmål:
    - Hent relevante noder
    - Vurder relevans
    - Generer strukturert svar med sitater
    - Verifiser sitatene mot nodene (men på en mer effektiv måte)
    """
    _emit(f"Worker answers the subquery \"{state['subquery'].subquery}\"", event="info")

    try:
        retriever = state["retriever"]
        question = state["subquery"].subquery
        severity = state.get("query_severity", "Green")
        
        empathy_instruction = ""
        #if severity in ("Yellow", "Red"):
        empathy_instruction = (
            "Hvis brukeren beskriver noe vanskelig. Anerkjenn at dette kan oppleves tøft "
            "før du gir informasjon. Vis empati, men bare basert på det som faktisk "
            "er relevant for spørsmålet.\n"
        )

        # Retrieval
        nodes = retriever.retrieve(question) or []
        _emit(f"Retrieved {len(nodes)} nodes", event="info")

        # Debug: emit the exact nodes this worker retrieved (text + score),
        # so a test can record the precise source set used for the answer.
        # With similarity_top_k <= MAX_NODES_FOR_CONTEXT this is identical to
        # the context fed to the LLM. Off unless explicitly requested.
        if state.get("debug_emit_nodes"):
            node_dump = [
                {
                    "score": float(getattr(n, "score", 0.0) or 0.0),
                    "url": (_node_meta(n).get("url") or ""),
                    "node_type": _node_meta(n).get("node_type", ""),
                    "text": _node_text(getattr(n, "node", n)),
                }
                for n in nodes
            ]
            _emit(json.dumps(node_dump, ensure_ascii=False), event="retrieved_nodes")

        thresholds = state.get("relevancy_thresholds", {
            "strong": 0.60,
            "medium": 0.55,
            "weak": 0.35,
        })

        if not nodes:
            state["subquery"].response_validity = "not valid"
            state["subquery"].answer = (
                "Jeg har dessverre ikke informasjon om dette i kildene jeg har tilgang til."
            )
            return {"completed_subqueries": [state["subquery"]],
                    "input_tokens": 0,
                    "output_tokens": 0}

        # 2) Relevans / band
        best_nws = max(nodes, key=lambda n: n.score)
        best_score = float(getattr(best_nws, "score", 0.0))
        band = _classify_relevancy(best_score, thresholds)


        refs: List[Reference] = []
        seen_urls = set()

        for nws in nodes:
            node_obj = getattr(nws, "node", nws)
            meta = getattr(node_obj, "metadata", {}) or {}

            url = (meta.get("url") or "").strip()
            if not url:
                continue

            if url in seen_urls:
                continue

            seen_urls.add(url)
            refs.append({
                "name": (meta.get("title") or "Ingen tittel").lstrip(),
                "url": url,
                "icon_url": meta.get("icon_url", ""),
                "relevancy_index": float(getattr(nws, "score", 0.0)),
            })

            if len(refs) >= 5:
                break

        state["subquery"].response_validity_index = best_score

        if band == "Rejected":
            state["subquery"].response_validity = "not valid"
            state["subquery"].answer = (
                "Jeg har dessverre ikke informasjon om dette i kildene jeg har tilgang til."
            )
            return {"completed_subqueries": [state["subquery"]],
                    "input_tokens": 0,
                    "output_tokens": 0}

        # 3) Begrens hvor mange noder vi bruker videre
        #    - få noder gir mye mindre prompt + raskere sitat-sjekk
        #    - garanter minst én artikkel i konteksten hvis en kvalifiserer
        original_top = nodes[:MAX_NODES_FOR_CONTEXT]
        nodes_for_context = _ensure_article_in_top(nodes, MAX_NODES_FOR_CONTEXT)
        nodes_for_verification = _ensure_article_in_top(nodes, MAX_NODES_FOR_VERIFICATION)

        if any(_node_meta(n).get("node_type") == "article" for n in nodes_for_context) \
                and not any(_node_meta(n).get("node_type") == "article" for n in original_top):
            _emit("Article promoted into top-N (was qa-only by score)", event="info")

        ctx = _format_context_from_nodes(nodes_for_context)
        prompt_value = GROUNDED_PROMPT.format(
            question=question,
            context=ctx,
            empathy_hint=empathy_instruction,
        )

        ga, in_tokens, out_tokens = _invoke_with_usage(
            state["llm"].with_structured_output(GroundedAnswer),
            prompt_value,
        )

        # logging.info(
        #     f"Subquery '{question}' brukte ca. {in_tokens} input tokens, {out_tokens} output tokens"
        # )

        # 4) Claims-verifisering – gjør den litt billigere
        #    a) Hvis du vil være raskere: dropp fuzzy (sett fuzzy_min_ratio=None)
        #    b) Eller behold den, men med færre noder (vi bruker nodes_for_verification)
        results = _verify_claims(
            ga,
            nodes_for_verification,
            min_quote_chars=8,
            collapse_whitespace=True,
            case_sensitive=False,
            fuzzy_min_ratio=60,    # sett til None for enda mer fart
        )

        global_problems = results.get("global_problems", [])
        claims_report = results.get("claims_report", [])

        # Entailment-gate: downgrade claims whose (real) quote doesn't actually
        # support them. Adds at most ONE small batched LLM call per subquery,
        # and only when the cheap term-overlap pre-filter flags something.
        ent_in = ent_out = 0
        if state.get("entailment_check", True):
            ent_in, ent_out = _apply_entailment_gate(claims_report, state["llm"])

        answer_wrapped = _wrap_at_nearest_space(ga.answer, width=120)

        # 5) validering og UI-output med detaljer
        _emit(f"## Delspørsmål: {question}", event="systeminfo")
        _emit("## Svar på delspørsmål:", event="systeminfo")
        _emit(answer_wrapped, event="systeminfo")
        _emit("\u00A0\n", event="systeminfo")
        _emit(" --- ", event="systeminfo")
        _emit("## Validering av påstander:", event="systeminfo")

        state["subquery"].response_validity = "valid"

        for claim_entry in claims_report:
            idx = claim_entry["claim_index"]
            claim_text = claim_entry["claim_text"]
            any_citation_valid = claim_entry["any_citation_valid"]
            all_citations_valid = claim_entry["all_citations_valid"]
            problems = claim_entry["problems"]
            citations_report = claim_entry["citations_report"]

            _emit("\n", event="systeminfo")
            _emit(f"# **Påstand {idx + 1}: {claim_text}** ", event="systeminfo")
            _emit(f"Minst én sitat-treff: {any_citation_valid}", event="systeminfo")
            _emit(f"Alle sitater gyldige: {all_citations_valid}", event="systeminfo")

            if problems:
                _emit("**Problemer for denne påstanden:**", event="systeminfo")
                for p in problems:
                    _emit(f"- {p}", event="systeminfo")

            _emit("Sitat-tilknytning:", event="systeminfo")

            for cit in citations_report:
                cit_i = cit["citation_index"]
                found_in_nodes = cit["found_in_nodes"]
                urls = cit["matched_node_urls"]

                url_str = ""
                for u in urls:
                    url_val = u or "Ingen URL"
                    url_str += f"[{url_val}]({url_val}) \n"

                quote_val = cit["quote"] or ""
                short_quote = quote_val.strip()
                if len(short_quote) > 140:
                    short_quote = short_quote[:137] + "..."
                short_quote = _wrap_at_nearest_space(short_quote, width=120)

                def esc(cell: str) -> str:
                    return cell.replace("|", "\\|")

                s = f"{cit_i} {'✅' if found_in_nodes else '❌'}  {short_quote} \n {esc(url_str)}"
                _emit(s, event="systeminfo")

            any_cit_problem = any(cit["problems"] for cit in citations_report)
            # --------------------------------------
            # streng validering: hvis noen sitater har problemer, forkast hele svaret
            #
            # if any_cit_problem:
            #     _emit("**Detaljer per sitat:**", event="systeminfo")
            #     for cit in citations_report:
            #         if not cit["problems"]:
            #             continue
            #         cit_i = cit["citation_index"]
            #         _emit(f"- Sitat {cit_i}:", event="systeminfo")
            #         for cp in cit["problems"]:
            #             _emit(f"  - {cp}", event="systeminfo")
            #     state["subquery"].response_validity = "not valid"
            
            
            
            
            # ---------------------------------------
            # mykere validering: vis sitat-problemer, men forkast ikke hele svaret – vurder andelen gyldige claims
            #
            if any_cit_problem:
                _emit("**Detaljer per sitat:**", event="systeminfo")
                for cit in citations_report:
                    if not cit["problems"]:
                        continue
                    cit_i = cit["citation_index"]
                    _emit(f"- Sitat {cit_i}:", event="systeminfo")
                    for cp in cit["problems"]:
                        _emit(f"  - {cp}", event="systeminfo")

                # Ikke forkast hele svaret – vurder andelen gyldige claims
                valid_claims = sum(1 for c in claims_report if c["any_citation_valid"])
                total_claims = len(claims_report)
                claims_threshold = state.get("claims_valid_threshold", 1.0)

                if total_claims == 0 or valid_claims == 0:
                    # Ingen claims kunne støttes i det hele tatt
                    state["subquery"].response_validity = "not valid"
                elif valid_claims / total_claims < claims_threshold:
                    # Færre enn terskelen av claims er støttet
                    state["subquery"].response_validity = "not valid"
                else:
                    # Majoriteten er støttet – behold svaret
                    state["subquery"].response_validity = "valid"
                    _emit(
                        f"⚠️ {total_claims - valid_claims} av {total_claims} claims mangler sitat, "
                        f"men {valid_claims}/{total_claims} er gyldige – svaret beholdes.",
                        event="systeminfo"
                    )

            _emit("\u00A0\n", event="systeminfo")
            _emit(" --- ", event="systeminfo")

        res = state["subquery"].response_validity
        _emit(f"## Resultat: {res}", event="systeminfo")


        state["subquery"].answer = ga.answer
        state["subquery"].short_answer = ga.short_answer
        state["subquery"].references = refs

        return {
            "completed_subqueries": [state["subquery"]],
            "input_tokens": in_tokens + ent_in,
            "output_tokens": out_tokens + ent_out,
        }

    except Exception as e:
        logging.error("Failed to execute query_grounded: %s", e)
        state["subquery"].response_validity = "not valid"
        state["subquery"].answer = "Jeg klarte ikke å verifisere sitatene nå."
        return {
            "completed_subqueries": [state["subquery"]],
            "input_tokens": 0,
            "output_tokens": 0,
        }



def synthesizer(state: State_Answer) -> Dict[str, Any]:
    """Sette sammen endelig svar basert på del-svarene."""
    
    _emit( "Synthesize full answer", event="info")

    llm = state["llm"]

    try:
        sq: List[SubQuery] = state.get("completed_subqueries", []) or []

        completed_report_answers = ""
        completed_report_answers_non_valid = ""

        for s in sq:
            if s.response_validity == "valid":
                combined = f"Subquery: {s.subquery}\n\nAnswer: {s.answer}\n\n"
                completed_report_answers += combined
            #else:
                #completed_report_answers_non_valid += (
                #    f'\n\nBeklager, men jeg kunne ikke svare på spørsmålet: "{s.subquery}"'
                #)

        if completed_report_answers:
            logging.info("Aggregating these answers: %s", completed_report_answers)

            messages = [
                SystemMessage(content=(
                    "Du er en vennlig, empatisk og kunnskapsrik helseveileder laget for å hjelpe ungdom i Norge (alder 13–19 år).\n\n"
                    "Din oppgave er å sette sammen et enkelt, helhetlig og sammenhengende svar på norsk (bokmål) "
                    "**kun basert på den gitte listen med del-svar**.\n"
                    "Du skal **ikke finne på eller legge til ny informasjon, påstander, forklaringer "
                    "eller råd** som ikke uttrykkelig finnes i den gitte teksten.\n"
                    "Hvis noe mangler, er uklart eller motsier seg selv, skal du bare utelate det.\n\n"
                    "Målet ditt:\n"
                    "- Slå sammen overlappende eller gjentatte poenger fra de gitte del-svarene til et ryddig og lettlest sammendrag.\n"
                    "- Behold innhold og tone slik de er skrevet.\n"
                    "- Ikke legg til nye påstander, tolkninger eller veiledning.\n\n"
                    "Tone og stil:\n"
                    "- Rolig, vennlig og støttende tone. Du kan formulere overganger naturlig.\n"
                    "- Vis gjerne at du forstår at temaet kan være vanskelig, uten å legge til ny informasjon.\n"
                    "- Klart språk, korte setninger, ungdomsvennlig.\n"
                    "- Unngå faguttrykk hvis de ikke allerede står i teksten.\n\n"
                    "Formatering:\n"
                    "- Bruk korte overskrifter når det hjelper på lesbarheten.\n"
                    "- Ikke referer eksplisitt til kilder eller 'del-svar'.\n"
                    "- Ikke fyll på med fraser som 'Her er...' eller 'Nedenfor finner du...'.\n\n"
                    "Hvis del-svarene ikke inneholder brukbar informasjon, skal du svare:\n"
                    "\"Det vet jeg ikke basert på kildene.\""
                )),
                HumanMessage(
                    content=f"Here is the list of answers: {completed_report_answers}"
                ),
            ]

            aggregated_answer, in_tokens, out_tokens = _invoke_with_usage(llm, messages)

            ref_list: List[Reference] = []
            for s in sq:
                if s.references and s.response_validity == "valid":
                    ref_list.extend(s.references)

            top5 = _dedupe_references(ref_list, top_k=5)

            final_short = ""
            for s in sq:
                if s.response_validity == "valid" and s.short_answer:
                    final_short = s.short_answer
                    break

            return {
                "validate_response_result": "Accepted",
                "final_answer": aggregated_answer.content + completed_report_answers_non_valid,
                "final_short_answer": final_short,
                "references": top5,
                "input_tokens": in_tokens,     
                "output_tokens": out_tokens,   
            }

        # Ingen gyldige del-svar
        placeholder = _pick_cannot_answer_placeholder(state.get("query_severity", "Green"))
        return {
            "validate_response_result":"Rejected",
            "final_answer": placeholder,
            "final_short_answer": placeholder, 
            "references": [],
        }

    except Exception as e:
        logging.error("Failed to execute synthesizer: %s", e)
        return {
            "final_answer": (
                "Jeg beklager, men jeg klarte ikke å sette sammen et helhetlig svar nå."
            ),
            "references": [],
        }

def pick_response_style(severity: str, stance: str) -> str:
    """Auto-rute fra (severity, stance) til en av RESPONSE_STYLES.

    Deterministisk policy-lookup, ikke en LLM-call. Brukes når klienten
    ikke har sendt en eksplisitt response_style i request body.

      Red                            → crisis
      affected_party + Yellow        → supportive
      affected_party + Green         → warm
      info_seeker + Yellow           → warm
      info_seeker / ambiguous + Green → factual (skip rewrite)
    """
    sev = (severity or "").strip()
    st = (stance or "").strip()
    if sev == "Red":
        return "crisis"
    if st == "affected_party":
        return "supportive" if sev == "Yellow" else "warm"
    if sev == "Yellow":
        return "warm"
    return "factual"


def apply_response_style(state: State_Answer) -> Dict[str, Any]:
    """Omskriv det endelige svaret i riktig stil.

    Stilen bestemmes av (i) klient-override i state['response_style'],
    eller (ii) auto-routing fra severity + stance via pick_response_style.
    'factual' hopper over LLM-callen helt. 'warm'/'supportive'/'crisis'
    bruker hver sin prompt fra registry.
    """
    # Bypass hvis svaret allerede er rejected (placeholder, ingen kilder)
    if state.get("validate_response_result") == "Rejected":
        _emit("Skipping style rewrite: response was Rejected", event="info")
        return {}

    answer = state.get("final_answer", "")
    if not answer or len(answer) < 50:
        return {}

    # Resolve effective style: klient-override hvis gyldig, ellers auto.
    override = (state.get("response_style") or "").strip()
    if override in RESPONSE_STYLES:
        style = override
        source = "override"
        _emit(f"Response style: {style} (client override)", event="info")
    else:
        style = pick_response_style(
            state.get("query_severity", ""),
            state.get("stance", ""),
        )
        source = "auto"
        _emit(f"Response style: {style} (auto)", event="info")

    # Safety floor: Red severity tvinger alltid 'crisis', også når klienten
    # har overstyrt til noe annet. Et test-/eval-valg av f.eks. 'factual' på
    # et Red-spørsmål ville gitt et tørt fakta-svar på en akutt krise – det
    # er aldri riktig. Auto-routingen returnerer allerede 'crisis' på Red,
    # så denne sjekken er bare relevant når en klient-override forsøker å
    # gå rundt det.
    if state.get("query_severity") == "Red" and style != "crisis":
        _emit(
            f"Response style: forced crisis (was {style}, severity=Red)",
            event="info",
        )
        style = "crisis"
        source = "forced_red"

    # Factual = skip rewrite, behold svaret som det er.
    if style == "factual":
        return {"response_style": "factual", "response_style_source": source}

    prompt_template = _STYLE_TO_PROMPT.get(style)
    if prompt_template is None:
        # Ukjent stil — burde ikke skje, men ikke krasj.
        logging.warning("Unknown response style %r, skipping rewrite", style)
        return {"response_style": style, "response_style_source": source}

    try:
        prompt_value = prompt_template.format(answer=answer)
        rewritten, in_tokens, out_tokens = _invoke_with_usage(
            state["llm"], prompt_value
        )
        return {
            "final_answer": rewritten.content,
            "response_style": style,
            "response_style_source": source,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
        }
    except Exception as e:
        logging.error("apply_response_style (%s) failed: %s", style, e)
        return {"response_style": style}  # behold originalt svar ved feil

def emit_query_answer_references(state: State_Answer) -> Dict[str, Any]:
    """Emitter endelig svar + referanser som events."""

    _emit( "Aggregating the final answer", event="info")

    try:
        completed = state.get("completed_subqueries") or []
        scores = [
            float(getattr(sq, "response_validity_index", 0.0) or 0.0)
            for sq in completed
        ]
        best_node_score = max(scores) if scores else 0.0
        relevancy_band = _classify_relevancy(
            best_node_score,
            {"strong": 0.60, "medium": 0.55, "weak": 0.35},
        ) if scores else ""

        status_payload = json.dumps(
            {
                "refined_query": state.get("refined_query", ""),
                "query_severity": state.get("query_severity", ""),
                "stance": state.get("stance", ""),
                "harm_to_others_tense": state.get("harm_to_others_tense", "na"),
                "response_style": state.get("response_style", ""),
                "response_style_source": state.get("response_style_source", ""),
                "relevancy_band": relevancy_band,
                "best_node_score": best_node_score,
                "validate_response_result": state.get("validate_response_result", ""),
            },
            ensure_ascii=False,
        )
        _emit(status_payload, event="query_status")

        q = state.get("refined_query")

        _emit(q, event="Refined query")

        _emit(f"\n## Du spurte\n{state['refined_query']}\n")
        _emit("\n## Svar\n")

        answer = state.get("final_answer", "")
        short_answer = state.get("final_short_answer", "")

        for line in answer.splitlines(True):
            _emit(line, event = "answer")
        _emit("\n", event = "answer")
        
        for line in short_answer.splitlines(True):
            _emit(line, event = "short_answer")
        _emit("\n", event = "short_answer")

        top5 = state["references"]

        if top5:
            for r in top5:
                name = r.get("name", "Uten tittel")
                url = r.get("url", "#")
                icon_url = r.get("icon_url")

                if icon_url:
                    bullet = f'[{name}]({url}) ||IMG|| {icon_url}\n'
                else:
                    bullet = f'[{name}]({url})\n'
                _emit(bullet, event="references")
                
        usage_payload = json.dumps(
            {
                "input_tokens": state.get("input_tokens", 0),
                "output_tokens": state.get("output_tokens", 0),
            },
            ensure_ascii=False,
        )
        _emit(usage_payload, event="Token usage")
        
        # priser i USD per 1M tokens
        price_input_usd_per_m = float(os.environ["PRICE_INPUT_USD_PER_M"])
        price_output_usd_per_m = float(os.environ["PRICE_OUTPUT_USD_PER_M"])
        USD_TO_NOK = 10  # grovt anslag

        input_tokens = state.get("input_tokens", 0) or 0
        output_tokens = state.get("output_tokens", 0) or 0

        kost_usd = (
            input_tokens * price_input_usd_per_m
            + output_tokens * price_output_usd_per_m
        ) / 1_000_000

        kost_nok = kost_usd * USD_TO_NOK

        _emit(f"\nKost: {kost_nok:.4f} NOK", event="Token usage")
        
    except Exception as e:
        logging.error("Failed to execute emit_query_answer_references: %s", e)
        return {
            "final_answer": (
                "Jeg beklager, men jeg klarte ikke å sette sammen et helhetlig svar nå."
            ),
            "references": [],
        }

    return {}


def assign_workers(state: State_Answer) -> List[Send]:
    """Opprett en worker for hver delspørring."""

    _emit( "Assign a worker to each section in the plan", event="info")
    
    return [
        Send(
            "query_grounded",
            {
                "subquery": s,
                "query_engine": state["query_engine"],
                "similarity_cutoff": state["similarity_cutoff"],
                "llm": state["llm"],
                "retriever": state["retriever"],
                "conversation_str": state.get("conversation_str", ""),
                "query_severity": state.get("query_severity", "Green"),
                "claims_valid_threshold": state.get("claims_valid_threshold", 1.0),
                "entailment_check": state.get("entailment_check", True),
                "debug_emit_nodes": state.get("debug_emit_nodes", False),
            },
        )
        for s in state["subqueries"]
    ]

def related_queries(state: State_Answer) -> dict:

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
        
    return {"related_queries": related_queries}

# def related_queries_dialog_from_query(state: State_Answer) -> dict:
#     _emit("Find dialog-driven related queries (query contains full history)", event="info")

#     llm = state["llm"]
    
#     conversation_str = state.get("conversation_str")
#     last_query = state.get("refined_query")
      
#     print(f'<<<history: {conversation_str}>>>')
#     if not conversation_str:
#         _emit("[]", event="related queries")
#         return {"related_queries": []}

#     # (valgfritt) kutt litt for kost: siste ~8000 tegn
#     conversation_str = conversation_str[-8000:]

#     # 1) Plan: last question + next intents
#     plan = _make_dialog_plan(llm, conversation_str)
#     last_user_q = (plan.last_user_question or "").strip()
#     intent_texts = [i.intent for i in (plan.intents or [])][:4]
    
#     print(f'\n----------\n<<<plan:{plan}\nlast_user_q:{last_user_q}\nintent_texts:{intent_texts}>>>')

#     # 2) Retrieve kandidater per intent
#     retriever = _build_related_queries_retriever(
#         index_qa_bank=state["index_related_queries"],
#         top_k=10,
#         cutoff=0.0,  # hent bredt, la rerank bestemme
#         query_severity=state.get("query_severity"),
#         main_category=state.get("main_category"),
#     )

#     candidates = {}  # node_id -> candidate
#     for intent in intent_texts:
#         results = retriever.retrieve(intent) or []
#         for r in results:
#             node = getattr(r, "node", r)
#             meta = getattr(node, "metadata", {}) or {}
#             node_id = getattr(node, "node_id", getattr(node, "id_", "")) or ""
#             text = _node_text(node).strip()
#             if not node_id or not text:
#                 continue

#             # hard-exclude: ikke foreslå nesten samme som siste brukerspørsmål
#             if last_user_q and partial_ratio(_normalize(text), _normalize(last_user_q)) > 92:
#                 continue

#             if node_id not in candidates:
#                 doc_id = meta.get("from_doc_id", node_id)
#                 candidates[node_id] = {
#                     "id": str(doc_id),
#                     "node_id": str(node_id),
#                     "text": text,
#                     "severity": meta.get("severity", ""),
#                 }

#     cand_list = list(candidates.values())
#     if not cand_list:
#         _emit("[]", event="related queries")
#         return {"related_queries": []}

#     # 3) Rerank på “naturlig fortsettelse”
#     cand_list = cand_list[:24]
#     cand_block = "\n".join([f"- node_id={c['node_id']}\n  q={c['text']}" for c in cand_list])

#     rerank_prompt = (
#         "Du skal velge hvilke kandidatspørsmål fra en spørsmålsbank som er den mest NATURLIGE "
#         "fortsettelsen i dialogen.\n\n"
#         "Gi score 0–1 basert på:\n"
#         "- Dialogfit: naturlig neste steg gitt historikken\n"
#         "- Ikke repetisjon av siste brukerspørsmål\n"
#         "- Fremdrift: avklaring/tiltak/risiko/når søke hjelp\n"
#         "- Unngå duplikater\n\n"
#         f"Siste brukerspørsmål: {last_user_q!r}\n\n"
#         f"Samtalehistorikk:\n{conversation_str}\n\n"
#         f"Kandidatspørsmål:\n{cand_block}\n"
#     )

#     reranked: RerankResult = llm.with_structured_output(RerankResult).invoke(rerank_prompt)
#     score_map = {x.node_id: x.score for x in (reranked.ranked or [])}
#     cand_list.sort(key=lambda c: score_map.get(c["node_id"], 0.0), reverse=True)

#     # 4) Velg maks 3 med enkel diversitet
#     picked = []
#     used_norm = set()
#     for c in cand_list:
#         if len(picked) >= 3:
#             break
#         norm = _normalize(c["text"])
#         if any(partial_ratio(norm, u) > 90 for u in used_norm):
#             continue
#         picked.append(c)
#         used_norm.add(norm)

#     related_queries = [
#         {"keyword": p.get("severity", ""), "query": p["text"], "node_id": p["node_id"]}
#         for p in picked
#     ]

#     _emit(json.dumps(related_queries, ensure_ascii=False), event="related queries")
#     return {"related_queries": related_queries}


def related_queries_dialog_from_query(state: State_Answer) -> dict:
    _emit("Related queries: single LLM selection (history-aware)", event="info")

    llm = state["llm"]
    conversation_history = (state.get("conversation_history") or "").strip()
    last_q = (state.get("refined_query") or state.get("query") or "").strip()

    # 1) Hent kandidater raskt (uten intents)
    retriever = _build_related_queries_retriever(
        index_qa_bank=state["index_related_queries"],
        top_k=30,          # hent litt bredt, men ikke for mye
        cutoff=0.0,        # la LLM velge
        query_severity=state.get("query_severity"),
        main_category=state.get("main_category"),
    )

    # Du kan bruke last_q for retrieval (vanligvis best). 
    results = retriever.retrieve(last_q) or []

    # 2) Pakk kandidatene i en enkel liste
    candidates = []
    for r in results:
        node = getattr(r, "node", r)
        meta = getattr(node, "metadata", {}) or {}
        node_id = getattr(node, "node_id", getattr(node, "id_", "")) or ""
        text = _node_text(node).strip()
        if not node_id or not text:
            continue

        # Hard-exclude: ikke foreslå nesten identisk med siste spørsmål
        if last_q and partial_ratio(_normalize(text), _normalize(last_q)) > 92:
            continue

        candidates.append({
            "node_id": str(node_id),
            "text": text,
            "severity": meta.get("severity", ""),
            "category": meta.get("category", ""),
            "score": float(getattr(r, "score", 0.0) or 0.0),
        })

    # De-dupe på node_id og begrens
    seen = set()
    uniq = []
    for c in candidates:
        if c["node_id"] in seen:
            continue
        seen.add(c["node_id"])
        uniq.append(c)

    uniq = uniq[:24]  # limit for tokens

    if not uniq:
        _emit("[]", event="related queries")
        return {"related_queries": []}

    candidates_jsonl = "\n".join(json.dumps(x, ensure_ascii=False) for x in uniq)

    # 3) ÉN LLM-call: velg topp 3 basert på historikk + siste spørsmål
    prompt = (
        "Du hjelper ungdom i Norge. Du får samtalehistorikk, siste brukerspørsmål, "
        "og en liste med kandidatspørsmål fra en spørsmålsbank.\n\n"
        "Oppgave:\n"
        "- Velg MAKS 2 kandidatspørsmål som er en NATURLIG fortsettelse i dialogen.\n"
        "- Ikke velg kandidater som bare gjentar siste spørsmål.\n"
        "- Velg kandidater som flytter dialogen videre (avklaring, neste steg, risiko, når søke hjelp).\n"
        "- IKKE endre teksten i kandidatene. Du skal bare returnere node_id.\n"
        "- Hvis ingen passer, returner tom liste.\n\n"
        f"Samtalehistorikk:\n{conversation_history}\n\n"
        f"Siste spørsmål:\n{last_q}\n\n"
        f"Kandidatspørsmål (JSONL):\n{candidates_jsonl}\n"
    )

    selection: RelatedSelection = llm.with_structured_output(RelatedSelection).invoke(prompt)

    selected_ids = selection.selected_node_ids[:2] if selection.selected_node_ids else []
    selected_map = {c["node_id"]: c for c in uniq}

    picked = [selected_map[i] for i in selected_ids if i in selected_map]

    related_queries = [
        {"keyword": p.get("severity", ""), "query": p["text"], "node_id": p["node_id"]}
        for p in picked
    ]

    _emit(json.dumps(related_queries, ensure_ascii=False), event="related queries")
    return {"related_queries": related_queries}

# def related_queries_dialog_from_query(state: State_Answer) -> dict:
#     _emit("Related queries: no-LLM heuristic ranking", event="info")

#     conversation_history = (state.get("conversation_history") or "").strip()
#     last_q = (state.get("refined_query") or state.get("query") or "").strip()

#     retriever = _build_related_queries_retriever(
#         index_qa_bank=state["index_related_queries"],
#         top_k=25,
#         cutoff=0.0,  # hent bredt
#         query_severity=state.get("query_severity"),
#         main_category=state.get("main_category"),
#     )

#     results = retriever.retrieve(last_q) or []

#     # Heuristisk scoring:
#     # - start med embedding score
#     # - straff nesten identisk med last_q
#     # - bonus for "progression" ord (valgfritt) / variasjon
#     progression_terms = [
#         "hva kan jeg gjøre", "hva gjør jeg", "hva bør jeg", "når bør", "bør jeg oppsøke",
#         "hvor kan jeg få hjelp", "hva er normalt", "risiko", "symptomer", "tegn", "råd", "tips"
#     ]
#     prog_norm = [_normalize(t) for t in progression_terms]

#     candidates = []
#     for r in results:
#         node = getattr(r, "node", r)
#         meta = getattr(node, "metadata", {}) or {}
#         node_id = getattr(node, "node_id", getattr(node, "id_", "")) or ""
#         text = _node_text(node).strip()
#         if not node_id or not text:
#             continue

#         base = float(getattr(r, "score", 0.0) or 0.0)
#         norm_text = _normalize(text)
#         norm_last = _normalize(last_q)

#         # penalty for near-duplicate of last question
#         dup_pen = 0.0
#         if last_q:
#             sim = partial_ratio(norm_text, norm_last)
#             if sim > 92:
#                 continue  # hard drop
#             dup_pen = sim / 200.0  # e.g. 0.0–0.46

#         # progression bonus (lightweight)
#         prog_bonus = 0.0
#         if any(t in norm_text for t in prog_norm):
#             prog_bonus = 0.05

#         score = base + prog_bonus - dup_pen

#         candidates.append({
#             "node_id": str(node_id),
#             "text": text,
#             "severity": meta.get("severity", ""),
#             "score": score,
#         })

#     # sort
#     candidates.sort(key=lambda x: x["score"], reverse=True)

#     # diversitet: unngå nesten duplikater i de 3 du plukker
#     picked = []
#     used = []
#     for c in candidates:
#         if len(picked) >= 3:
#             break
#         n = _normalize(c["text"])
#         if any(partial_ratio(n, u) > 90 for u in used):
#             continue
#         picked.append(c)
#         used.append(n)

#     related_queries = [
#         {"keyword": p.get("severity", ""), "query": p["text"], "node_id": p["node_id"]}
#         for p in picked
#     ]

#     _emit(json.dumps(related_queries, ensure_ascii=False), event="related queries")
#     return {"related_queries": related_queries}
# ---------------------------------------------------------
# Bygg workflow
# ---------------------------------------------------------

builder = StateGraph(State_Answer)

builder.add_node("analyze_query", analyze_query)
builder.add_node("fast_single", fast_single)

builder.add_node("orchestrator", orchestrator)
builder.add_node("query_grounded", query_grounded)
builder.add_node("synthesizer", synthesizer, join=True)
builder.add_node("emit_query_answer_references", emit_query_answer_references)
builder.add_node("related_queries_dialog_from_query", related_queries_dialog_from_query)
builder.add_node("apply_response_style", apply_response_style)
builder.add_node("refuse_harm_to_others", refuse_harm_to_others)
builder.add_node("help_after_harm", help_after_harm)

# Start → analyse
builder.add_edge(START, "analyze_query")

# Etter analyse: refusal, help-after-harm, fasttrack eller full multisteg-flyt
builder.add_conditional_edges(
    "analyze_query",
    route_after_analysis,
    ["fast_single", "orchestrator", "refuse_harm_to_others", "help_after_harm"],
)

# Hvis multi: bruk eksisterende flyt
builder.add_conditional_edges("orchestrator", assign_workers, ["query_grounded"])
builder.add_edge("query_grounded", "synthesizer")
builder.add_edge("synthesizer", "apply_response_style")
builder.add_edge("apply_response_style", "emit_query_answer_references")

# Hvis fasttrack: gå rett til emitter
builder.add_edge("fast_single", "apply_response_style")
# (apply_response_style → emit_query_answer_references er allerede koblet)

# Refusal og help-after-harm: skip retrieval og apply_response_style. Begge
# nodene produserer ferdig formulert tekst (LLM-drevet med statisk
# fallback), og style-prompts er designet for å omskrive et grounded svar
# – ikke et refusal/help-svar som allerede har riktig tone.
builder.add_edge("refuse_harm_to_others", "emit_query_answer_references")
builder.add_edge("help_after_harm", "emit_query_answer_references")

# Videre er likt for begge ruter
builder.add_edge("emit_query_answer_references", "related_queries_dialog_from_query")
builder.add_edge("related_queries_dialog_from_query", END)

answer_workflow = builder.compile()
#from graph_utils import save_mermaid_diagram
#save_mermaid_diagram(answer_workflow.get_graph())