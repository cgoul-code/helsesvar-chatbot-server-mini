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
    PREJUDICE_ANSWER,
    PREJUDICE_SHORT_ANSWER,
    SELF_HARM_ANSWER,
    SELF_HARM_SHORT_ANSWER,
    REFUSE_HARM_PROMPT,
    HELP_AFTER_HARM_PROMPT,
    ADDRESS_PREJUDICE_PROMPT,
    SELF_HARM_PROMPT,
    HJELPETJENESTER_KATALOG,
    HJELPETJENESTER_BY_ID,
    format_hjelpetjeneste_linje,
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
    # Cheaper/faster model for auxiliary calls (analyze, orchestrate,
    # entailment, related-query selection). Falls back to `llm`.
    fast_llm: Any

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
    stance: Literal["info_seeker", "affected_party", "harm_to_others", "harm_to_self", "expresses_prejudice", "ambiguous"]
    harm_to_others_tense: Literal["planning", "completed", "unclear", "na"]
    # Brukerens eget kjønn, avledet i analyze_query. 'ukjent' = standard/ingen
    # antakelse. Brukes til å vinkle svaret der kjønn er relevant (prevensjon,
    # kropp, helse) uten å tilskrive brukeren feil kjønn.
    asker_gender: Literal["jente", "gutt", "ukjent"]
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
    
    # Tokens fra HOVEDmodellen (state["llm"], f.eks. gpt-4.1-mini).
    input_tokens: Annotated[int, add]
    output_tokens: Annotated[int, add]
    # Tokens fra FAST-modellen (state["fast_llm"], f.eks. gpt-4o-mini). Holdes
    # adskilt fordi den prises billigere – cost-beregningen priser hver modell
    # for seg i stedet for å bruke én felles pris (som ville vært feil).
    fast_input_tokens: Annotated[int, add]
    fast_output_tokens: Annotated[int, add]

    ''' tuning '''
    # Min fraction of cited claims that must be supported for an answer to
    # stay "valid" in query_grounded. Default 1.0.
    claims_valid_threshold: float
    # When True, query_grounded runs an LLM entailment gate that downgrades
    # claims whose (real) quote doesn't support them. Default True.
    entailment_check: bool

    # True once the final answer has been streamed token-by-token to the
    # client, so emit_query_answer_references doesn't re-emit it.
    answer_streamed: bool

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
    stance: Literal["info_seeker", "affected_party", "harm_to_others", "harm_to_self", "expresses_prejudice", "ambiguous"] = Field(
        default="info_seeker",
        description=(
            "Brukerens rolle i situasjonen: 'info_seeker' (ber om generell info), "
            "'affected_party' (beskriver noe som rammer brukeren selv), "
            "'harm_to_others' (ber om hjelp til å påføre andre skade — straffbart "
            "eller åpenbart skadelig), 'harm_to_self' (uttrykker tanker om eller "
            "intensjon til å skade seg selv — selvskading, selvmord o.l.), "
            "'expresses_prejudice' (uttrykker en fordom eller nedvurderende "
            "holdning mot en gruppe), eller 'ambiguous' (uklart hvem som er aktør)."
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
    main_category: str = Field(
        default="",
        description="Kort emne-stikkord. Fylles kun når needs_subqueries=True.",
    )
    asker_gender: Literal["jente", "gutt", "ukjent"] = Field(
        default="ukjent",
        description=(
            "Brukerens EGET kjønn, avledet kun når det går tydelig fram av "
            "brukerens egne ord. 'ukjent' er standard og skal brukes når "
            "kjønnet ikke er eksplisitt — ikke gjett. Gjelder brukeren selv, "
            "ikke andre personer som nevnes."
        ),
    )
    subqueries: List[str] = Field(
        default_factory=list,
        description=(
            "Delspørsmål når needs_subqueries=True (ellers tom liste). Hvert "
            "element er ett omskrevet delspørsmål på norsk bokmål."
        ),
    )

class WorkerState(TypedDict):
    subquery: SubQuery
    similarity_cutoff: float
    query_engine: BaseQueryEngine
    retriever: BaseRetriever
    llm: Any
    fast_llm: Any
    conversation_str: str
    query_severity: str
    asker_gender: str
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

def _chunk_text(chunk: Any) -> str:
    """Pull plain text out of a streamed message chunk.

    Azure/OpenAI chunks expose `.content` as a str; Anthropic can return a
    list of content blocks. Normalise both to a string.
    """
    content = getattr(chunk, "content", "") or ""
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(p.get("text", "") or "")
            else:
                parts.append(str(p))
        return "".join(parts)
    return content


def _stream_with_usage(llm, messages, event: str = "answer") -> tuple[str, int, int]:
    """Stream an LLM response, emitting each piece as it arrives, and return
    (full_text, input_tokens, output_tokens).

    This is what makes the answer appear token-by-token in the client instead
    of all at once after the whole graph has finished.
    """
    callback = UsageMetadataCallbackHandler()
    parts: List[str] = []
    for chunk in llm.stream(messages, config={"callbacks": [callback]}):
        piece = _chunk_text(chunk)
        if piece:
            parts.append(piece)
            _emit(piece, event=event)
    in_tok, out_tok = _extract_usage_tokens(callback.usage_metadata)
    return "".join(parts), in_tok, out_tok


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


# Standard NOK-per-USD når MERVERDI/kurs ikke er satt i env. Grovt anslag –
# overstyr med USD_TO_NOK i miljøet for en mer presis omregning.
_DEFAULT_USD_TO_NOK = 10.0


def _compute_cost(
    input_tokens: int,
    output_tokens: int,
    fast_input_tokens: int = 0,
    fast_output_tokens: int = 0,
) -> Dict[str, float]:
    """Regn ut kostnad, priset PER MODELL.

    Hoved- og fast-modell prises hver for seg, fordi de typisk har ulik pris
    (f.eks. gpt-4.1-mini vs gpt-4o-mini). Å bruke én felles pris ville vært
    feil. Priser leses fra env (USD per 1M tokens) med trygge defaults:
      - PRICE_INPUT_USD_PER_M / PRICE_OUTPUT_USD_PER_M           → hovedmodell
      - PRICE_FAST_INPUT_USD_PER_M / PRICE_FAST_OUTPUT_USD_PER_M → fast-modell
        (faller tilbake til hovedmodellens pris hvis ikke satt)
    En manglende env-variabel gir pris 0, ikke kræsj. Returnerer USD + NOK og
    totale token-tall (hoved + fast).
    """
    main_in = int(input_tokens or 0)
    main_out = int(output_tokens or 0)
    fast_in = int(fast_input_tokens or 0)
    fast_out = int(fast_output_tokens or 0)

    price_in = float(os.getenv("PRICE_INPUT_USD_PER_M", "0") or 0)
    price_out = float(os.getenv("PRICE_OUTPUT_USD_PER_M", "0") or 0)
    fprice_in = float(os.getenv("PRICE_FAST_INPUT_USD_PER_M", str(price_in)) or price_in)
    fprice_out = float(os.getenv("PRICE_FAST_OUTPUT_USD_PER_M", str(price_out)) or price_out)
    usd_to_nok = float(os.getenv("USD_TO_NOK", str(_DEFAULT_USD_TO_NOK)) or _DEFAULT_USD_TO_NOK)

    cost_usd = (
        main_in * price_in + main_out * price_out
        + fast_in * fprice_in + fast_out * fprice_out
    ) / 1_000_000

    return {
        "input_tokens": main_in + fast_in,
        "output_tokens": main_out + fast_out,
        "main_input_tokens": main_in,
        "main_output_tokens": main_out,
        "fast_input_tokens": fast_in,
        "fast_output_tokens": fast_out,
        "cost_usd": cost_usd,
        "cost_nok": cost_usd * usd_to_nok,
    }


def _emit_query_status(
    state: State_Answer,
    cost: Dict[str, float],
    relevancy_band: str,
    best_node_score: float,
) -> None:
    """Bygg og emit query_status-payloaden (info-ruten i klienten).

    Samlet i én helper så den kan emittes både fra emit_query_answer_references
    (svar-fasen) og på nytt fra related_queries-noden med den FULLE kostnaden
    (inkl. related_queries-kallet, som kjører etter den første emitteringen).
    """
    payload = json.dumps(
        {
            "refined_query": state.get("refined_query", ""),
            "query_severity": state.get("query_severity", ""),
            "stance": state.get("stance", ""),
            "harm_to_others_tense": state.get("harm_to_others_tense", "na"),
            "asker_gender": state.get("asker_gender", "ukjent"),
            "response_style": state.get("response_style", ""),
            "response_style_source": state.get("response_style_source", ""),
            "relevancy_band": relevancy_band,
            "best_node_score": best_node_score,
            "validate_response_result": state.get("validate_response_result", ""),
            "input_tokens": cost["input_tokens"],
            "output_tokens": cost["output_tokens"],
            "cost_usd": cost["cost_usd"],
            "cost_nok": cost["cost_nok"],
        },
        ensure_ascii=False,
    )
    _emit(payload, event="query_status")

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
            matched_texts: List[str] = []
            seen_texts: set = set()
            for n in matched_nodes_for_cit:
                try:
                    matched_urls.append(n.metadata.get("url", "Ingen URL"))
                except Exception:
                    matched_urls.append("Ingen URL")
                # Keep the full source passage so the entailment gate can resolve
                # pronouns / elliptical references ("Det avgjøres av...") that the
                # bare quote leaves dangling. Dedupe + cap to keep prompts small.
                node_txt = (_node_text(n) or "").strip()
                if node_txt and node_txt not in seen_texts:
                    seen_texts.add(node_txt)
                    matched_texts.append(node_txt[:1500])

            citations_report.append({
                "citation_index": cit_i,
                "url": cit.url,
                "quote": cit.quote,
                "found_in_nodes": found_in_nodes,
                "problems": this_cit_problems,
                "matched_node_urls": matched_urls,
                "matched_node_texts": matched_texts,
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
def _make_subquery(text: str) -> SubQuery:
    """Wrap a plain subquery string in an (unanswered) SubQuery for the workers."""
    return SubQuery(
        subquery=text,
        answer="",
        short_answer="",
        references=[],
        response_validity="not valid",
        response_validity_index=0.0,
    )


def analyze_query(state: State_Answer) -> Dict[str, Any]:
    """Renskriver spørsmålet, klassifiserer, OG (ved behov) dekomponerer i
    delspørsmål — alt i ett LLM-kall. Erstatter det tidligere separate
    orchestrator-kallet."""

    _emit("Analyze and possibly rewrite user query", event="info")

    llm = state.get("fast_llm") or state["llm"]

    conversation_str = state.get("conversation_str", "")
    original_q = state["query"]

    prompt = ANALYZE_QUERY_PROMPT.format(
        conversation_str=conversation_str,
        original_q=original_q,
    )

    planner = llm.with_structured_output(QueryPlan)
    plan, in_tokens, out_tokens = _invoke_with_usage(planner, prompt)

    # harm_to_others_tense er bare meningsfull for stance='harm_to_others'.
    # LLM-en setter den av og til (f.eks. 'planning') også for harm_to_self
    # eller andre stances; normaliser til 'na' så routing/UI/tester ikke ser
    # en tense som ikke gjelder.
    harm_tense = plan.harm_to_others_tense if plan.stance == "harm_to_others" else "na"

    # Vi skriver om query i state til renskrevet versjon
    print(f'Severity: {plan.query_severity}, Stance: {plan.stance}, Needs subqueries: {plan.needs_subqueries}, Harm tense: {harm_tense}, Gender: {plan.asker_gender}')

    # Bygg SubQuery-objekter fra delspørsmålene den samme callen produserte,
    # så vi slipper et eget orchestrator-kall (sparer ett round-trip).
    subqueries = [
        _make_subquery(text)
        for text in (plan.subqueries or [])
        if (text or "").strip()
    ]

    return {
        "refined_query": plan.refined_query,
        "needs_subqueries": plan.needs_subqueries,
        "query_severity": plan.query_severity,
        "stance": plan.stance,
        "harm_to_others_tense": harm_tense,
        "main_category": plan.main_category or "",
        "asker_gender": plan.asker_gender or "ukjent",
        "subqueries": subqueries,
        # analyze_query kjører på fast_llm → fast-tokens.
        "fast_input_tokens": in_tokens,
        "fast_output_tokens": out_tokens,
    }
    
def orchestrator(state: State_Answer) -> Dict[str, Any]:
    """DEPRECATED / ubrukt: delspørsmål genereres nå i analyze_query (samme
    LLM-kall), og route_after_analysis fan-er ut workerne direkte. Beholdt
    midlertidig for referanse; ikke koblet inn i grafen lenger."""
    
    _emit("Orchestrator that generates a plan for solving the question", event="info")

    try:
        llm = state.get("fast_llm") or state["llm"]
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
        "fast_llm": state.get("fast_llm") or state["llm"],
        "conversation_str": state.get("conversation_str", ""),
        "query_severity": state.get("query_severity", "Green"),
        "asker_gender": state.get("asker_gender", "ukjent"),
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
    fast_in_tokens = result.get("fast_input_tokens", 0) or 0
    fast_out_tokens = result.get("fast_output_tokens", 0) or 0

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
        # NB: token-feltene er Annotated[int, add] – returner kun DELTA-en for
        # dette steget (ikke state.get(...) + delta, som ville dobbelt-telt).
        # query_grounded skiller allerede hoved- (GROUNDED) og fast-tokens
        # (entailment); vi viderefører begge.
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "fast_input_tokens": fast_in_tokens,
        "fast_output_tokens": fast_out_tokens,
        "validate_response_result": validate_response_result,
    }
    
def route_after_analysis(state: State_Answer):
    """Bestem neste steg basert på stance, tense og needs_subqueries.

    Stance dominerer. For 'harm_to_others' deler vi i to grener basert på
    tense: 'completed' går til help_after_harm (skadebegrensning, ikke
    avvisning), alt annet går til refuse_harm_to_others. Dette unngår at
    kunnskapsbasen (som er skrevet for ofre/generell info) blir brukt til
    å besvare et gjerningsperson-spørsmål.

    Multistegs-ruten fan-er nå ut workere DIREKTE (returnerer Send-liste),
    siden delspørsmålene allerede er produsert i analyze_query. Det sparer
    det tidligere separate orchestrator-kallet.
    """
    # Selvskade/selvmord overstyrer alt annet: en safety-kritisk gren som
    # møter brukeren med krisestøtte i stedet for et RAG-svar.
    if state.get("stance") == "harm_to_self":
        return "respond_self_harm"
    if state.get("stance") == "harm_to_others":
        tense = state.get("harm_to_others_tense", "unclear")
        if tense == "completed":
            return "help_after_harm"
        return "refuse_harm_to_others"
    if state.get("stance") == "expresses_prejudice":
        return "address_prejudice"
    if state.get("needs_subqueries") and (state.get("subqueries") or []):
        return assign_workers(state)  # List[Send] → query_grounded
    return "fast_single"


# Enkelte fagtilbud er så situasjonsspesifikke at de aldri bør falle ut av et
# harm-svar. LLM-seleksjonen i harm-grenene er instruksjonsdrevet og kan bomme;
# disse deterministiske reglene garanterer at riktig tilbud kommer med. Vi er
# allerede inne i harm_to_others her, så aktøren er brukeren – derfor holder det
# å oppdage situasjonstypen i spørsmålet/historikken.

# Brukerens egen (mulige) seksuelle krenkelse/grenseoverskridelse (kontakt/
# atferd, ikke bare bildedeling) → Tryggprat. Rene bildesaker dekkes av
# Slettmeg-regelen under, så bilde-substantiver ligger ikke her.
_SEXUAL_HARM_PATTERNS = re.compile(
    r"(krenk\w*|overgrep\w*|voldt\w*|misbruk\w*|seksuell?\w*|seksuelt|"
    r"blott\w*|tafs\w*|befølt?\w*|grenseoverskrid\w*|"
    r"ufrivillig\s+sex|tvang\w*\s+til\s+sex)",
    re.IGNORECASE,
)

# Bilder/film/private opplysninger involvert → Slettmeg.no (fjerne innhold).
# Krever et innholds-substantiv (ikke bare et delings-verb), så «spredte
# rykter» o.l. ikke feilutløser tilbudet.
_IMAGE_SHARING_PATTERNS = re.compile(
    r"(nakenbild\w*|nudes?|sexbild\w*|bilde\w*|bilder|video\w*|film\w*|"
    r"filmet|skjermbild\w*|opptak|klipp|"
    r"private\s+opplysning\w*|privat\s+info)",
    re.IGNORECASE,
)

# Pågående/umiddelbar fare akkurat nå → Politiet (akutt fare) / 112.
_ACUTE_DANGER_PATTERNS = re.compile(
    r"(akkurat\s+nå|pågår|holder\s+på\s+(med\s+)?å|umiddelbar\s+fare|"
    r"skal\s+til\s+å|er\s+i\s+fare|i\s+livsfare|"
    r"truer\s+med\s+å\s+(drepe|skade|ta\s+livet)|"
    r"kommer\s+til\s+å\s+(skade|drepe))",
    re.IGNORECASE,
)


def _matches(pattern: re.Pattern, *texts: str) -> bool:
    blob = " ".join(t for t in texts if t)
    return bool(pattern.search(blob))


def _ensure_service_in_answer(answer: str, service_id: str, blurb: str = "") -> str:
    """Garanter at et bestemt tilbud er nevnt i svaret; injiser hvis ikke.

    Hopper over hvis tilbudet allerede er nevnt – på navn, domene eller
    telefonnummer (sifre) – så vi ikke dupliserer det LLM-en eventuelt
    allerede tok med.
    """
    svc = HJELPETJENESTER_BY_ID.get(service_id)
    if not svc or not answer:
        return answer
    navn = (svc.get("navn") or "").lower()
    nett = (svc.get("nettside") or "").lower().replace("www.", "")
    nett_ok = bool(nett) and not nett.startswith("ingen")
    tlf_digits = re.sub(r"\D", "", svc.get("telefon") or "")
    haystack = answer.lower()
    answer_digits = re.sub(r"\D", "", answer)
    already = (
        (navn and navn in haystack)
        or (nett_ok and nett in haystack)
        or (len(tlf_digits) >= 3 and tlf_digits in answer_digits)
    )
    if already:
        return answer
    linje = format_hjelpetjeneste_linje(svc, blurb=blurb)
    if not linje:
        return answer
    return answer.rstrip() + "\n" + linje


# (predikat, service_id, blurb) – evalueres i rekkefølge. Blurb tom => bruk
# tilbudets egen «Når relevant»-tekst (forkortet).
_HARM_SERVICE_RULES = [
    (
        _SEXUAL_HARM_PATTERNS,
        "tryggprat",
        "hjelp og veiledning hvis du er bekymret for egne seksuelle tanker "
        "eller handlinger, eller har gjort noe over grensa mot en annen",
    ),
    (
        _IMAGE_SHARING_PATTERNS,
        "slettmeg",
        "hjelp til å få fjernet bilder eller private opplysninger som er "
        "spredt på nett",
    ),
    (
        _ACUTE_DANGER_PATTERNS,
        "politiet-akutt",
        "ring umiddelbart hvis noen er i akutt fare akkurat nå",
    ),
]


def _inject_specialized_harm_services(answer: str, *context_texts: str) -> str:
    """Legg til situasjonsspesifikke fagtilbud som ikke bør utelates."""
    for pattern, service_id, blurb in _HARM_SERVICE_RULES:
        if _matches(pattern, *context_texts):
            answer = _ensure_service_in_answer(answer, service_id, blurb)
    return answer


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
            tjenester_katalog=HJELPETJENESTER_KATALOG,
        )
        result, in_tokens, out_tokens = _invoke_with_usage(
            llm.with_structured_output(RefusalResponse),
            prompt_value,
        )
        final_answer = _inject_specialized_harm_services(
            result.answer, query, conversation_str
        )
        return {
            "final_answer": final_answer,
            "final_short_answer": result.short_answer,
            "references": [],
            "validate_response_result": "Accepted",
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
        }
    except Exception as e:
        logging.error("refuse_harm_to_others LLM call failed, using static fallback: %s", e)
        return {
            "final_answer": _inject_specialized_harm_services(
                HARM_REFUSAL_ANSWER, query, conversation_str
            ),
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
            tjenester_katalog=HJELPETJENESTER_KATALOG,
        )
        result, in_tokens, out_tokens = _invoke_with_usage(
            llm.with_structured_output(RefusalResponse),
            prompt_value,
        )
        final_answer = _inject_specialized_harm_services(
            result.answer, query, conversation_str
        )
        return {
            "final_answer": final_answer,
            "final_short_answer": result.short_answer,
            "references": [],
            "validate_response_result": "Accepted",
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
        }
    except Exception as e:
        logging.error("help_after_harm LLM call failed, using static fallback: %s", e)
        return {
            "final_answer": _inject_specialized_harm_services(
                HELP_AFTER_HARM_ANSWER, query, conversation_str
            ),
            "final_short_answer": HELP_AFTER_HARM_SHORT_ANSWER,
            "references": [],
            "validate_response_result": "Accepted",
            "input_tokens": 0,
            "output_tokens": 0,
        }


def address_prejudice(state: State_Answer) -> Dict[str, Any]:
    """Svar når brukeren uttrykker en fordom/holdning mot en gruppe.

    Egen gren (parallelt med harm-nodene) fordi et rent RAG-svar bare kan
    gjengi det retrieval henter — for «jeg liker ikke homofile» henter det
    typisk «aksepter deg selv»-artikler og feiltolker brukeren som om hen
    strever med egen identitet. Denne noden resonnerer fritt: møter ubehaget
    uten å validere fordommen, og slår fast andres likeverd og rett til å
    være den de er. LLM-drevet, faller tilbake til statisk melding ved feil.
    """
    _emit("Stance=expresses_prejudice: address_prejudice node", event="info")

    llm = state["llm"]
    query = state.get("refined_query") or state.get("query") or ""
    conversation_str = state.get("conversation_str", "") or ""

    try:
        prompt_value = ADDRESS_PREJUDICE_PROMPT.format(
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
        logging.error("address_prejudice LLM call failed, using static fallback: %s", e)
        return {
            "final_answer": PREJUDICE_ANSWER,
            "final_short_answer": PREJUDICE_SHORT_ANSWER,
            "references": [],
            "validate_response_result": "Accepted",
            "input_tokens": 0,
            "output_tokens": 0,
        }


def respond_self_harm(state: State_Answer) -> Dict[str, Any]:
    """Krise-svar når brukeren uttrykker fare for SEG SELV (selvskading,
    selvmordstanker o.l.) — stance=harm_to_self.

    Egen gren på linje med harm-nodene: ikke et RAG-svar, men et varmt
    krisesvar som anerkjenner brukeren, formidler håp og loser videre til
    krisehjelp. Garanterer deterministisk at Mental Helse Ungdom (116 123,
    døgnåpen) er med. LLM-drevet, faller tilbake til statisk melding ved feil.
    """
    _emit("Stance=harm_to_self: respond_self_harm node", event="info")

    llm = state["llm"]
    query = state.get("refined_query") or state.get("query") or ""
    conversation_str = state.get("conversation_str", "") or ""

    try:
        prompt_value = SELF_HARM_PROMPT.format(
            query=query,
            conversation_str=conversation_str,
            tjenester_katalog=HJELPETJENESTER_KATALOG,
        )
        result, in_tokens, out_tokens = _invoke_with_usage(
            llm.with_structured_output(RefusalResponse),
            prompt_value,
        )
        # Mental Helse Ungdom skal aldri falle ut av et selvskade-svar.
        final_answer = _ensure_service_in_answer(
            result.answer,
            "mental-helse-ungdom",
            "ring eller chat hvis tankene blir for tunge – åpen hele døgnet",
        )
        return {
            "final_answer": final_answer,
            "final_short_answer": result.short_answer,
            "references": [],
            "validate_response_result": "Accepted",
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
        }
    except Exception as e:
        logging.error("respond_self_harm LLM call failed, using static fallback: %s", e)
        return {
            "final_answer": SELF_HARM_ANSWER,
            "final_short_answer": SELF_HARM_SHORT_ANSWER,
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
    "Du er en presis faktasjekker. For hver PÅSTAND får du ett eller flere SITAT "
    "(hentet ordrett fra kildene) og KILDETEKSTEN sitatet er klippet fra. "
    "Avgjør om kilden FAKTISK STØTTER påstanden — altså om den handler om SAMME "
    "tiltak/forhold og logisk medfører påstanden.\n"
    "- Sitatet er bare et utdrag. Bruk KILDETEKSTEN til å forstå sammenhengen og "
    "til å tolke pronomen og henvisninger i sitatet (f.eks. «Det», «den», «slik»). "
    "Hvis kildeteksten gjør det klart hva sitatet viser til, og den støtter "
    "påstanden, så er den supported=true — selv om det løsrevne sitatet alene "
    "virker uklart.\n"
    "- Marker supported=false bare når kilden egentlig handler om noe ANNET "
    "(annet produkt/prevensjonsmiddel, annen aldersgruppe, annen ordning) eller "
    "motsier påstanden — ikke fordi det korte sitatet mangler kontekst.\n"
    "- Vær spesielt streng på forveksling av ulike, men beslektede ting "
    "(f.eks. p-stav vs p-plaster vs p-pille).\n"
    "Returner for hver indeks om den er supported (true/false).\n\n"
    "Påstander, sitater og kildetekst:\n"
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
    candidates: List[Tuple[Dict[str, Any], List[str], List[str]]] = []
    for ce in claims_report:
        if not ce.get("any_citation_valid"):
            continue
        quotes: List[str] = []
        sources: List[str] = []
        seen_src: set = set()
        for c in ce.get("citations_report", []):
            if not c.get("found_in_nodes"):
                continue
            q = (c.get("quote") or "").strip()
            if q:
                quotes.append(q)
            # Collect the source passage(s) the quote was matched in, so the LLM
            # can resolve context-dependent quotes instead of judging them blind.
            for txt in (c.get("matched_node_texts") or []):
                txt = (txt or "").strip()
                if txt and txt not in seen_src:
                    seen_src.add(txt)
                    sources.append(txt)
        if not quotes:
            continue
        if not _entailment_needed(ce.get("claim_text", ""), quotes):
            continue
        candidates.append((ce, quotes, sources))

    if not candidates:
        return 0, 0

    lines = []
    for i, (ce, quotes, sources) in enumerate(candidates):
        joined = " | ".join(quotes)
        block = f"{i}. PÅSTAND: {ce.get('claim_text', '')}\n   SITAT: {joined}"
        if sources:
            joined_src = "\n   ---\n   ".join(sources)
            block += f"\n   KILDETEKST:\n   {joined_src}"
        lines.append(block)
    prompt = ENTAILMENT_PROMPT + "\n\n".join(lines)

    try:
        res, in_tok, out_tok = _invoke_with_usage(
            llm.with_structured_output(_EntailmentResult), prompt
        )
        supported = {v.index: v.supported for v in res.verdicts}
    except Exception as e:
        logging.error("Entailment batch call failed, keeping string-match validity: %s", e)
        return 0, 0

    for i, (ce, _quotes, _sources) in enumerate(candidates):
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
        asker_gender = state.get("asker_gender", "ukjent")

        empathy_instruction = ""
        #if severity in ("Yellow", "Red"):
        empathy_instruction = (
            "Hvis brukeren beskriver noe vanskelig. Anerkjenn at dette kan oppleves tøft "
            "før du gir informasjon. Vis empati, men bare basert på det som faktisk "
            "er relevant for spørsmålet.\n"
        )

        # Kjønns-hint: styrer kun vinkling/utvalg i svaret, ikke hvilke noder
        # som hentes. 'ukjent' (standard) skal holde svaret kjønnsnøytralt.
        if asker_gender == "jente":
            gender_instruction = (
                "Brukeren er selv jente/kvinne. Der kjønn er relevant (f.eks. "
                "prevensjon, kropp, helse), vinkle svaret ut fra dette. Ikke "
                "tilskriv brukeren et annet kjønn.\n"
            )
        elif asker_gender == "gutt":
            gender_instruction = (
                "Brukeren er selv gutt/mann. Der kjønn er relevant (f.eks. "
                "prevensjon, kropp, helse), vinkle svaret ut fra dette. Ikke "
                "tilskriv brukeren et annet kjønn.\n"
            )
        else:
            gender_instruction = (
                "Brukerens kjønn er ukjent. Ikke anta kjønn. Hold svaret "
                "kjønnsnøytralt, og dekk relevante perspektiver der kjønn "
                "ellers ville spilt inn.\n"
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
            gender_hint=gender_instruction,
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
            ent_in, ent_out = _apply_entailment_gate(
                claims_report, state.get("fast_llm") or state["llm"]
            )

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
            # GROUNDED-kallet bruker hovedmodellen; entailment-porten fast_llm.
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "fast_input_tokens": ent_in,
            "fast_output_tokens": ent_out,
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



# System-instruks for trofast sammenstilling av del-svar (nøytral tone).
# Brukes når stilen er 'factual', men det finnes flere del-svar som må slås
# sammen. For warm/supportive/crisis brukes de finjusterte stil-promptene i
# registry i stedet (de slår sammen og setter tone i samme omskrivning).
_SYNTH_FAITHFUL_SYSTEM = (
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
)


def synthesize_style_stream(state: State_Answer) -> Dict[str, Any]:
    """Slå sammen gyldige del-svar OG bruk riktig tone i ÉN streamet LLM-call.

    Erstatter de tidligere to nodene `synthesizer` + `apply_response_style`.
    Svaret streames token-for-token til klienten (event="answer") mens det
    genereres, slik at brukeren ser tekst med en gang i stedet for å vente på
    at hele grafen blir ferdig.
    """
    llm = state["llm"]

    sq: List[SubQuery] = state.get("completed_subqueries", []) or []
    valid = [s for s in sq if s.response_validity == "valid"]

    # Referanser + kort svar fra de gyldige del-svarene.
    ref_list: List[Reference] = []
    final_short = ""
    for s in valid:
        if s.references:
            ref_list.extend(s.references)
        if not final_short and s.short_answer:
            final_short = s.short_answer
    top5 = _dedupe_references(ref_list, top_k=5)

    # Ingen gyldige del-svar → placeholder. emit-noden skriver den ut.
    if not valid:
        placeholder = _pick_cannot_answer_placeholder(state.get("query_severity", "Green"))
        _emit("Ingen gyldige del-svar – returnerer placeholder", event="info")
        return {
            "validate_response_result": "Rejected",
            "final_answer": placeholder,
            "final_short_answer": placeholder,
            "references": [],
            "answer_streamed": False,
        }

    # Kildetekst = de gyldige svarene (uten Subquery-stillas; stil-promptene
    # forventer ren svartekst).
    source_answer = "\n\n".join(s.answer for s in valid if s.answer).strip()

    # Velg stil: klient-override > auto-routing, med Red-safety-floor.
    override = (state.get("response_style") or "").strip()
    if override in RESPONSE_STYLES:
        style, source_kind = override, "override"
    else:
        style = pick_response_style(state.get("query_severity", ""), state.get("stance", ""))
        source_kind = "auto"
    if state.get("query_severity") == "Red" and style != "crisis":
        style, source_kind = "crisis", "forced_red"

    _emit(f"Synthesize + style ({style}, source={source_kind}) — streaming", event="info")

    # Rask vei: 'factual' med ett enkelt del-svar trenger verken sammenslåing
    # eller omskrivning → stream råsvaret direkte uten LLM-call.
    if style == "factual" and len(valid) == 1:
        _emit(source_answer, event="answer")
        _emit("\n", event="answer")
        return {
            "validate_response_result": "Accepted",
            "final_answer": source_answer,
            "final_short_answer": final_short,
            "references": top5,
            "response_style": "factual",
            "response_style_source": source_kind,
            "answer_streamed": True,
        }

    # Bygg ÉN melding som både slår sammen og setter tone.
    prompt_template = _STYLE_TO_PROMPT.get(style)
    if prompt_template is not None:
        # Gjenbruk den finjusterte stil-prompten; mat den den sammenstilte
        # kildeteksten så sammenslåing + tone skjer i samme omskrivning.
        messages = prompt_template.format(answer=source_answer)
    else:
        # 'factual' med flere del-svar: trofast sammenstilling, nøytral tone.
        messages = [
            SystemMessage(content=_SYNTH_FAITHFUL_SYSTEM),
            HumanMessage(content=f"Her er listen med del-svar:\n\n{source_answer}"),
        ]

    try:
        full, in_tok, out_tok = _stream_with_usage(llm, messages, event="answer")
        _emit("\n", event="answer")
        return {
            "validate_response_result": "Accepted",
            "final_answer": full.strip() or source_answer,
            "final_short_answer": final_short,
            "references": top5,
            "response_style": style,
            "response_style_source": source_kind,
            "answer_streamed": True,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
        }
    except Exception as e:
        # Fallback: stream råsvaret slik at brukeren får noe brukbart.
        logging.error("synthesize_style_stream failed, streaming source answer: %s", e)
        _emit(source_answer, event="answer")
        _emit("\n", event="answer")
        return {
            "validate_response_result": "Accepted",
            "final_answer": source_answer,
            "final_short_answer": final_short,
            "references": top5,
            "response_style": style,
            "response_style_source": source_kind,
            "answer_streamed": True,
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

        # Kostnad så langt: analyze_query + svar-nodene (alt som har kjørt før
        # denne noden), priset per modell. related_queries-kallet kjører ETTER
        # denne emitteringen og folder inn den FULLE kostnaden via en ny
        # query_status fra related_queries-noden.
        cost = _compute_cost(
            state.get("input_tokens", 0),
            state.get("output_tokens", 0),
            state.get("fast_input_tokens", 0),
            state.get("fast_output_tokens", 0),
        )
        _emit_query_status(state, cost, relevancy_band, best_node_score)

        q = state.get("refined_query")

        _emit(q, event="Refined query")

        _emit(f"\n## Du spurte\n{state['refined_query']}\n")
        _emit("\n## Svar\n")

        answer = state.get("final_answer", "")
        short_answer = state.get("final_short_answer", "")

        # The answer is normally streamed token-by-token in
        # synthesize_style_stream. Only emit it here for paths that DON'T go
        # through that node (harm/prejudice nodes, or a placeholder), i.e.
        # when it hasn't already been streamed.
        if not state.get("answer_streamed"):
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
                "input_tokens": cost["input_tokens"],
                "output_tokens": cost["output_tokens"],
                "cost_usd": cost["cost_usd"],
                "cost_nok": cost["cost_nok"],
            },
            ensure_ascii=False,
        )
        _emit(usage_payload, event="Token usage")
        _emit(f"\nKost: {cost['cost_nok']:.4f} NOK", event="Token usage")

        # Lagre i state så related_queries-noden kan re-emitte query_status med
        # den fulle kostnaden uten å regne relevans på nytt.
        return {
            "best_node_score": best_node_score,
            "relevancy_band": relevancy_band,
        }

    except Exception as e:
        logging.error("Failed to execute emit_query_answer_references: %s", e)
        return {
            "final_answer": (
                "Jeg beklager, men jeg klarte ikke å sette sammen et helhetlig svar nå."
            ),
            "references": [],
        }


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
                "fast_llm": state.get("fast_llm") or state["llm"],
                "retriever": state["retriever"],
                "conversation_str": state.get("conversation_str", ""),
                "query_severity": state.get("query_severity", "Green"),
                "asker_gender": state.get("asker_gender", "ukjent"),
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

    llm = state.get("fast_llm") or state["llm"]
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

    selection, rq_in_tokens, rq_out_tokens = _invoke_with_usage(
        llm.with_structured_output(RelatedSelection), prompt
    )

    selected_ids = selection.selected_node_ids[:2] if selection.selected_node_ids else []
    selected_map = {c["node_id"]: c for c in uniq}

    picked = [selected_map[i] for i in selected_ids if i in selected_map]

    related_queries = [
        {"keyword": p.get("severity", ""), "query": p["text"], "node_id": p["node_id"]}
        for p in picked
    ]

    _emit(json.dumps(related_queries, ensure_ascii=False), event="related queries")

    # related_queries kjører på fast_llm. Re-emit query_status med den FULLE
    # kostnaden (svar-fasen i state + dette kallet), så panel-tallet i klienten
    # dekker hele agent-kjøringen. state har ennå ikke fått reducer'ens add av
    # disse tokenene, så vi legger dem til lokalt her.
    full_cost = _compute_cost(
        state.get("input_tokens", 0),
        state.get("output_tokens", 0),
        (state.get("fast_input_tokens", 0) or 0) + rq_in_tokens,
        (state.get("fast_output_tokens", 0) or 0) + rq_out_tokens,
    )
    _emit_query_status(
        state,
        full_cost,
        state.get("relevancy_band", "") or "",
        state.get("best_node_score", 0.0) or 0.0,
    )

    return {
        "related_queries": related_queries,
        "fast_input_tokens": rq_in_tokens,
        "fast_output_tokens": rq_out_tokens,
    }

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

builder.add_node("query_grounded", query_grounded)
# Slår sammen tidligere synthesizer + apply_response_style til én streamet
# node. join=True (defer): kjøres sist, etter at alle query_grounded-workere
# er ferdige, og fungerer også når fast_single er eneste forgjenger.
builder.add_node("synthesize_style_stream", synthesize_style_stream, join=True)
builder.add_node("emit_query_answer_references", emit_query_answer_references)
builder.add_node("related_queries_dialog_from_query", related_queries_dialog_from_query)
builder.add_node("refuse_harm_to_others", refuse_harm_to_others)
builder.add_node("help_after_harm", help_after_harm)
builder.add_node("address_prejudice", address_prejudice)
builder.add_node("respond_self_harm", respond_self_harm)

# Start → analyse
builder.add_edge(START, "analyze_query")

# Etter analyse: refusal, help-after-harm, fasttrack eller multisteg-fan-out.
# Multisteg fan-er ut workere direkte fra route_after_analysis (Send-liste),
# så det trengs ingen egen orchestrator-node lenger.
builder.add_conditional_edges(
    "analyze_query",
    route_after_analysis,
    ["fast_single", "query_grounded", "refuse_harm_to_others", "help_after_harm", "address_prejudice", "respond_self_harm"],
)

# Hvis multi: workere → samle+stream → emit
builder.add_edge("query_grounded", "synthesize_style_stream")
builder.add_edge("synthesize_style_stream", "emit_query_answer_references")

# Hvis fasttrack: samme samle+stream-node (ett gyldig del-svar)
builder.add_edge("fast_single", "synthesize_style_stream")

# Refusal og help-after-harm: skip retrieval og synthesize_style_stream. Begge
# nodene produserer ferdig formulert tekst (LLM-drevet med statisk
# fallback), og style-prompts er designet for å omskrive et grounded svar
# – ikke et refusal/help-svar som allerede har riktig tone. Disse emittes
# (ikke-streamet) i emit_query_answer_references siden answer_streamed=False.
builder.add_edge("refuse_harm_to_others", "emit_query_answer_references")
builder.add_edge("help_after_harm", "emit_query_answer_references")
builder.add_edge("address_prejudice", "emit_query_answer_references")
builder.add_edge("respond_self_harm", "emit_query_answer_references")

# Videre er likt for begge ruter
builder.add_edge("emit_query_answer_references", "related_queries_dialog_from_query")
builder.add_edge("related_queries_dialog_from_query", END)

answer_workflow = builder.compile()
#from graph_utils import save_mermaid_diagram
#save_mermaid_diagram(answer_workflow.get_graph())