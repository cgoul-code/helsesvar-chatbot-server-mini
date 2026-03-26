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

from llama_index.core.base.response.schema import Response
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

from registry import subqueries_prompt, GROUNDED_PROMPT, CANNOT_ANSWER_PLACEHOLDER, EMPATHY_REWRITE_PROMPT

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


# Hvis best_score > dette → vi anser treffet som veldig godt og kan gjøre mindre arbeid
HIGH_CONFIDENCE_THRESHOLD = 0.82

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
    relevancy_band: str
    best_node_score: float
    response: Response | None
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

class WorkerState(TypedDict):
    subquery: SubQuery
    similarity_cutoff: float
    query_engine: BaseQueryEngine
    retriever: BaseRetriever
    llm: Any
    conversation_str: str  
    query_severity: str     
    

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
    
    prompt = (
        "Du er en helseveileder som hjelper ungdom i Norge.\n\n"
        "Oppgave:\n"
        "1) Skriv brukerens siste spørsmål om til én kort, tydelig og konkret formulering "
        "på norsk, uten å endre meningen.\n"
        
        "2) Vurder om spørsmålet bør deles opp i flere delspørsmål for å gi et godt svar.\n"
        "- Sett needs_subqueries = True KUN hvis spørsmålet inneholder 3+ helt separate temaer.\n"
        "- For de fleste spørsmål er False riktig.\n"
        
        "3) Kategoriser alvorlighetsgraden av spørsmålet i én av tre kategorier: \"Green\", \"Yellow\", eller \"Red\".\n\n"
        "ALVORLIGHESGRAD \"GREEN\":\n"
        "- Forebyggende og trygghetsskapende.\n"
        "- Spørsmål som ber om generell informasjon, kunnskap eller veiledning for å forebygge problemer og styrke god seksuell helse.\n"
        "- Brukeren ønsker å øke forståelse, trygghet og bevissthet (f.eks. samtykke, prevensjon, kommunikasjon, følelser, kunnskap om kroppen).\n"
        "- Ingen akutt situasjon eller personlig krise beskrives.\n"
        "Eksempel: «Hvordan kan jeg snakke med partneren min om grenser?» eller «Hvilke typer prevensjon finnes?». \n\n"
        "ALVORLIGHESGRAD \"YELLOW\":\n"
        "- Utfordringer eller sårbare situasjoner.\n"
        "- Spørsmål som beskriver bekymringer, vansker eller risikoer som kan kreve refleksjon eller støtte, men som ikke er akutte eller umiddelbart farlige.\n"
        "- Kan innebære vanskelige følelser, usikkerhet i relasjoner, uønskede opplevelser eller behov for råd utover generell informasjon.\n"
        "- Brukeren kan ha behov for hjelp eller veiledning, men situasjonen regnes ikke som en akutt krise.\n"
        "Eksempel: «Hva bør jeg gjøre hvis partneren min ikke respekterer grensene mine?», "
        "«Jeg angrer på at jeg sendte et nakenbilde», eller temaer som «porno», «seksuelt press», «problemer med samtykke», «(ulovlige) fetisjer».\n\n"
        "ALVORLIGHESGRAD \"RED\":\n"
        "- Alvorlige eller akutte situasjoner.\n"
        "- Spørsmål som gjelder alvorlige hendelser eller kriser der personen kan være i fare eller ha betydelig risiko for skade.\n"
        "- Omfatter vold, overgrep, tvang, akutte psykiske kriser eller andre situasjoner som krever umiddelbar oppfølging eller profesjonell hjelp.\n"
        "Eksempel: «Stefaren min tvinger meg til å ha sex», «Hvor kan jeg finne barnepornografi?», «Jeg ble voldtatt i går».\n\n"
        "ALVORLIGHESGRAD settes i Severity"
        f"Brukeren har tidligere spurt:\n\"\"\"{conversation_str}\"\"\""
        f"Brukerens siste spørsmål:\n\"\"\"{original_q}\"\"\""
    )

    planner = llm.with_structured_output(QueryPlan)
    plan: QueryPlan = planner.invoke(prompt)
    

    # Vi skriver om query i state til renskrevet versjon
    return {
        "refined_query": plan.refined_query,
        "needs_subqueries": plan.needs_subqueries,
        "query_severity": plan.query_severity
    }
    
def orchestrator(state: State_Answer) -> Dict[str, Any]:
    """Planlegger – genererer delspørsmål basert på brukerens spørsmål."""
    
    _emit("Orchestrator that generates a plan for solving the question", event="info")

    try:
        llm = state["llm"]
        prompt = subqueries_prompt(query=state["refined_query"])

        planner = llm.with_structured_output(SubQueries)
        report_queries: SubQueries = planner.invoke(prompt)

        return {"subqueries": report_queries.subqueries}

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
    """Bestem neste node basert på needs_subqueries."""
    if state.get("needs_subqueries"):
        return "orchestrator"
    return "fast_single"
    
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

        # 1) Retrieval
        nodes = retriever.retrieve(question) or []

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
                "icon_url": meta.get("icon_url", "Ingen URL for ikon"),
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
        nodes_for_context = nodes[:MAX_NODES_FOR_CONTEXT]
        nodes_for_verification = nodes[:MAX_NODES_FOR_VERIFICATION]

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
            _emit(f"# **Påstand {idx}: {claim_text}** ", event="systeminfo")
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

                if total_claims == 0 or valid_claims == 0:
                    # Ingen claims kunne støttes i det hele tatt
                    state["subquery"].response_validity = "not valid"
                elif valid_claims / total_claims < 0.5:
                    # Færre enn halvparten av claims er støttet
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
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
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

def empathy_rewrite(state: State_Answer) -> Dict[str, Any]:
    """Postprosesser det endelige svaret for å gi det en varmere, mer empatisk tone."""

    _emit("Rewriting answer with empathy", event="info")
    
    # Bypass if response was rejected (placeholder answer, no valid sources)
    #
    if state.get("validate_response_result") == "Rejected":
        _emit("Skipping empathy rewrite: response was Rejected", event="info")
        return {}

    llm = state["llm"]
    answer = state.get("final_answer", "")
    severity = state.get("query_severity", "Green")

    # Ikke omskriv placeholder-svar
    if not answer or len(answer) < 50:
        return {}

    try:
        prompt_value = EMPATHY_REWRITE_PROMPT.format(
            answer=answer,
            severity=severity,   # <-- også fiks denne buggen: var hardkodet til "Red"
        )

        rewritten, in_tokens, out_tokens = _invoke_with_usage(llm, prompt_value)
        
        print(f"Rewritten answer: fra:<{answer}> +ntil--->\n<{rewritten.content}>")
        return {
            "final_answer": rewritten.content,
            "input_tokens": in_tokens,      
            "output_tokens": out_tokens,    
        }

    except Exception as e:
        logging.error("empathy_rewrite failed: %s", e)
        return {}  # behold originalt svar ved feil

def emit_query_answer_references(state: State_Answer) -> Dict[str, Any]:
    """Emitter endelig svar + referanser som events."""
    
    _emit( "Aggregating the final answer", event="info")

    try:
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
builder.add_node("empathy_rewrite", empathy_rewrite)

# Start → analyse
builder.add_edge(START, "analyze_query")

# Etter analyse: enten fasttrack eller full multisteg-flyt
builder.add_conditional_edges(
    "analyze_query",
    route_after_analysis,              # funksjonen over
    ["fast_single", "orchestrator"],   # mulig returverdier
)

# Hvis multi: bruk eksisterende flyt
builder.add_conditional_edges("orchestrator", assign_workers, ["query_grounded"])
builder.add_edge("query_grounded", "synthesizer")
builder.add_edge("synthesizer", "empathy_rewrite")
builder.add_edge("empathy_rewrite", "emit_query_answer_references")

# Hvis fasttrack: gå rett til emitter
builder.add_edge("fast_single", "empathy_rewrite")
builder.add_edge("empathy_rewrite", "emit_query_answer_references")

# Videre er likt for begge ruter
builder.add_edge("emit_query_answer_references", "related_queries_dialog_from_query")
builder.add_edge("related_queries_dialog_from_query", END)

answer_workflow = builder.compile()
from graph_utils import save_mermaid_diagram
#save_mermaid_diagram(answer_workflow.get_graph())