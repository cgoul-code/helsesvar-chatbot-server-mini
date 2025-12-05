import json
import logging
import re
import textwrap
import unicodedata
import heapq

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

from registry import subqueries_prompt, GROUNDED_PROMPT


# ---------------------------------------------------------
# Datamodeller og typer
# ---------------------------------------------------------

class Reference(TypedDict):
    name: str
    url: str
    relevancy_index: float


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


class SubQuery(BaseModel):
    subquery: str = Field(description="The subquery")
    answer: str = Field(description="Answer to the subquery")
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
    from_node_id: str
    similarity_cutoff: float
    similarity_top_k: int
    relevancy_cutoff: float
    
    ''' calculated params'''
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
    
    input_tokens: Annotated[int, add]
    output_tokens: Annotated[int, add]


class WorkerState(TypedDict):
    subquery: SubQuery
    similarity_cutoff: float
    query_engine: BaseQueryEngine
    retriever: BaseRetriever
    llm: Any
    
class RelatedQueryAgent:
    """
    Handles the special case:
      - from_related_q == True
      - from_node_id is set
    Fetches answer directly from index_related_queries and streams it.
    """

    def __init__(self, index_related_queries):
        self.index = index_related_queries

    def _get_node(self, node_id: str):
        """
        Look up the node in index_related_queries.

        This is just a stub – adapt it to whatever your index actually is.
        For example:
          - if it's a dict:  return self.index.get(node_id)
          - if it's a vector DB: search by metadata node_id
        """
        # EXAMPLE if index is a dict:
        return self.index.get(node_id)

    async def run(self, *, from_node_id: str, _emit):
        """
        _emit(event, data=None, **extra)
        should be the same function you use in your old agent to send SSE.
        """
        node = self._get_node(from_node_id)

        if not node:
            await _emit(
                event="answer",
                structured_answer_delta="Jeg fant dessverre ikke noe svar for dette spørsmålet.",
            )
            await _emit(event="done")
            return

        # Tilpass disse feltene til din faktiske datastruktur
        answer_text = (
            node.get("answer")
            or node.get("content")
            or node.get("text")
            or ""
        )

        if not answer_text.strip():
            await _emit(
                event="answer",
                structured_answer_delta="Jeg fant dessverre ikke noe svar for dette spørsmålet.",
            )
            await _emit(event="done")
            return

        # Her kan du velge å streame stykkevis hvis du vil
        # For enkelhet sender jeg alt i ett event
        await _emit(
            event="answer",
            structured_answer_delta=answer_text,
        )

        # Hvis du også vil sende referanser/metadata fra noden, gjør det her:
        # if "references" in node:
        #     await _emit(
        #         event="references",
        #         structured_answer_delta="\n".join(node["references"]),
        #     )

        await _emit(event="done")


_POSSIBLE_META_IDS = ("doc_id", "from_doc_id", "document_id", "source_id")


# ---------------------------------------------------------
# Små hjelpefunksjoner
# ---------------------------------------------------------
def _message(delta: str, event: str = "Answer") -> None:
    """Sender streaming-event til klient (UI)."""
    writer = get_stream_writer()
    writer({"event": event, "message": delta})
    if event == 'info':
        logging.info(f'event: {event}, structured_answer_delta: {delta}')
        
def _emit(delta: str, event: str = "systeminfo") -> None:
    """Sender streaming-event til klient (UI)."""
    writer = get_stream_writer()
    writer({"event": event, "structured_answer_delta": delta})


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


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _node_text(n: Any) -> str:
    """Hent tekst fra ulike node-typer (TextNode/Document)."""
    t = getattr(n, "text", None)
    if t:
        return t

    get_content = getattr(n, "get_content", None)
    if callable(get_content):
        return get_content(metadata_mode="all") or ""

    return getattr(n, "get_text", lambda: "")() or ""


_WS = re.compile(r"\s+")
_TRANSLATE = str.maketrans({
    "\u2018": "'",
    "\u2019": "'",
    "\u201C": '"',
    "\u201D": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u00A0": " ",
    "\u202F": " ",
})
_ZERO_WIDTH = dict.fromkeys(map(ord, ["\u200B", "\u200C", "\u200D", "\u2060"]), None)
_CONTROL_CHARS = dict.fromkeys(range(0x00, 0x20), None)


def _normalize(
    s: str,
    *,
    collapse_ws: bool = True,
    case_sensitive: bool = False,
) -> str:
    """Normaliser tekst for sammenligning/søk."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_TRANSLATE)
    s = s.translate(_ZERO_WIDTH)
    s = s.translate(_CONTROL_CHARS)
    if collapse_ws:
        s = _WS.sub(" ", s)
    return s if case_sensitive else s.casefold()


def _format_context_from_nodes(
    nodes: List[Any],
    max_chars_per_node: int = 3000,
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
        txt = txt[:max_chars_per_node]
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




def _as_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default


def _as_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return default


def _build_related_queries_retriever(
    index_qa_bank: VectorStoreIndex,
    *,
    top_k: int,
    cutoff: float,
    query_severity: Optional[str],
    main_category: Optional[str],
) -> BaseRetriever:
    top_k = _as_int(top_k, 5) or 5
    cutoff = _as_float(cutoff, 0.0) or 0.0

    if query_severity == "Green":
        allowed_sev = ["Green"]
    elif query_severity == "Yellow":
        allowed_sev = ["Green", "Yellow"]
    else:
        allowed_sev = ["Green", "Yellow", "Red"]

    filters_list: List[MetadataFilter] = [
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

    if main_category:
        if isinstance(main_category, str):
            filters_list.append(
                MetadataFilter(
                    key="category",
                    value=main_category,
                    operator=FilterOperator.EQ,
                )
            )
        else:
            filters_list.append(
                MetadataFilter(
                    key="category",
                    value=list(main_category),
                    operator=FilterOperator.IN,
                )
            )

    composite = MetadataFilters(filters=filters_list, condition="and")

    return index_qa_bank.as_retriever(
        similarity_top_k=top_k,
        similarity_cutoff=cutoff,
        filters=composite,
    )


# ---------------------------------------------------------
# Noder i grafen
# ---------------------------------------------------------

def orchestrator(state: State_Answer) -> Dict[str, Any]:
    """Planlegger – genererer delspørsmål basert på brukerens spørsmål."""
    
    _message("Orchestrator that generates a plan for solving the question", event="info")

    try:
        llm = state["llm"]
        prompt = subqueries_prompt(query=state["query"])

        planner = llm.with_structured_output(SubQueries)
        report_queries: SubQueries = planner.invoke(prompt)

        return {"subqueries": report_queries.subqueries}

    except Exception as e:
        logging.error("Failed to execute orchestrator: %s", e)
        return {
            "subqueries": []
        }


def query_grounded(state: WorkerState) -> Dict[str, Any]:
    """
    For hvert delspørsmål:
    - Hent relevante noder
    - Vurder relevans
    - Generer strukturert svar med sitater
    - Verifiser sitatene mot nodene
    """
    _message(f"Worker answers the subquery \"{state['subquery'].subquery}\"", event="info")

    try:
        retriever = state["retriever"]
        question = state["subquery"].subquery

        nodes = retriever.retrieve(question) or []

        thresholds = state.get("relevancy_thresholds", {
            "strong": 0.60,
            "medium": 0.55,
            "weak": 0.35,
        })

        if not nodes:
            state["subquery"].response_validity = "not valid"
            state["subquery"].answer = (
                "Jeg beklager, men jeg kan bare svare på spørsmål basert på den gitte konteksten."
            )
            return {"completed_subqueries": [state["subquery"]]}

        best_nws = max(nodes, key=lambda n: n.score)
        best_score = float(getattr(best_nws, "score", 0.0))
        band = _classify_relevancy(best_score, thresholds)
        logging.info("Band: %s for %r, best score: %.3f", band, question, best_score)

        refs: List[Reference] = []
        for nws in nodes[:5]:
            node_obj = getattr(nws, "node", nws)
            meta = getattr(node_obj, "metadata", {}) or {}
            refs.append({
                "name": (meta.get("title") or "Ingen tittel").lstrip(),
                "url": meta.get("url", "Ingen URL"),
                "relevancy_index": float(getattr(nws, "score", 0.0)),
            })
        print('<1>')
        state["subquery"].response_validity_index = best_score

        if band == "Rejected":
            state["subquery"].response_validity = "not valid"
            state["subquery"].answer = (
                "Jeg beklager, men jeg kan bare svare på spørsmål basert på den gitte konteksten."
            )
            return {"completed_subqueries": [state["subquery"]]}

        ctx = _format_context_from_nodes(nodes)
        print('<2>')
        usage_callback = UsageMetadataCallbackHandler()
        chain = (
            RunnableLambda(lambda _: {"question": question, "context": ctx})
            | GROUNDED_PROMPT
            | state["llm"].with_structured_output(GroundedAnswer)
        )
        # 3) Kjør kjeden, men send inn callback via config
        ga: GroundedAnswer = chain.invoke(
            {},
            config={"callbacks": [usage_callback]},
        )

        print(f'<3> {ga}')
        in_tokens, out_tokens = _extract_usage_tokens(usage_callback.usage_metadata)
        logging.info(
            f"Subquery '{question}' brukte ca. {in_tokens} input tokens, {out_tokens} output tokens"
        )

        
        results = _verify_claims(
            ga,
            nodes,
            min_quote_chars=8,
            collapse_whitespace=True,
            case_sensitive=False,
            fuzzy_min_ratio=60,
        )
        print('<4>')
        global_problems = results.get("global_problems", [])
        claims_report = results.get("claims_report", [])
        answer_wrapped = _wrap_at_nearest_space(ga.answer, width=120)
        print('<5>')
        # UI-output
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
            if any_cit_problem:
                _emit("**Detaljer per sitat:**", event="systeminfo")
                for cit in citations_report:
                    if not cit["problems"]:
                        continue
                    cit_i = cit["citation_index"]
                    _emit(f"- Sitat {cit_i}:", event="systeminfo")
                    for cp in cit["problems"]:
                        _emit(f"  - {cp}", event="systeminfo")
                state["subquery"].response_validity = "not valid"

            _emit("\u00A0\n", event="systeminfo")
            _emit(" --- ", event="systeminfo")

        res = state["subquery"].response_validity
        _emit(f"## Resultat: {res}", event="systeminfo")
        print('<6>')

        state["subquery"].answer = ga.answer
        state["subquery"].references = refs
        print(f'<7>')
        
        return {"completed_subqueries": [state["subquery"]],
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,}

    except Exception as e:
        logging.error("Failed to execute query_grounded: %s", e)
        state["subquery"].response_validity = "not valid"
        state["subquery"].answer = "Jeg klarte ikke å verifisere sitatene nå."
        return {"completed_subqueries": [state["subquery"]],
                "input_tokens": 0,
                "output_tokens": 0,}


def synthesizer(state: State_Answer) -> Dict[str, Any]:
    """Sette sammen endelig svar basert på del-svarene."""
    
    _message( "Synthesize full answer", event="info")
    print('<8>')
    llm = state["llm"]

    try:
        sq: List[SubQuery] = state.get("completed_subqueries", []) or []

        completed_report_answers = ""
        completed_report_answers_non_valid = ""

        for s in sq:
            if s.response_validity == "valid":
                combined = f"Subquery: {s.subquery}\n\nAnswer: {s.answer}\n\n"
                completed_report_answers += combined
            else:
                completed_report_answers_non_valid += (
                    f'\n\nBeklager, men jeg kunne ikke svare på spørsmålet: "{s.subquery}"'
                )

        if completed_report_answers:
            logging.info("Aggregating these answers: %s", completed_report_answers)

            aggregated_answer = llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "Du er en vennlig, empatisk og kunnskapsrik helseveileder laget for å hjelpe ungdom i Norge (alder 13–19 år).\n\n"
                            "Din oppgave er å sette sammen et enkelt, helhetlig og sammenhengende svar på norsk (bokmål) "
                            "**kun basert på den gitte listen med del-svar**.\n"
                            "Du skal **ikke finne på, omformulere eller legge til ny informasjon, påstander, forklaringer "
                            "eller råd** som ikke uttrykkelig finnes i den gitte teksten.\n"
                            "Hvis noe mangler, er uklart eller motsier seg selv, skal du bare utelate det.\n\n"
                            "Målet ditt:\n"
                            "- Slå sammen overlappende eller gjentatte poenger fra de gitte del-svarene til et ryddig og lettlest sammendrag.\n"
                            "- Behold innhold og tone slik de er skrevet.\n"
                            "- Ikke legg til nye påstander, tolkninger eller veiledning.\n\n"
                            "Tone og stil:\n"
                            "- Rolig, vennlig og støttende, men ikke legg til ny empati som ikke står der fra før.\n"
                            "- Klart språk, korte setninger, ungdomsvennlig.\n"
                            "- Unngå faguttrykk hvis de ikke allerede står i teksten.\n\n"
                            "Formatering:\n"
                            "- Bruk korte overskrifter når det hjelper på lesbarheten.\n"
                            "- Ikke referer eksplisitt til kilder eller 'del-svar'.\n"
                            "- Ikke fyll på med fraser som 'Her er...' eller 'Nedenfor finner du...'.\n\n"
                            "Hvis del-svarene ikke inneholder brukbar informasjon, skal du svare:\n"
                            "\"Det vet jeg ikke basert på kildene.\""
                        )
                    ),
                    HumanMessage(
                        content=f"Here is the list of answers: {completed_report_answers}"
                    ),
                ]
            )

            ref_list: List[Reference] = []
            for s in sq:
                if s.references and s.response_validity == "valid":
                    for r in s.references:
                        ref_list.append(r)

            best_by_url: Dict[str, Reference] = {}
            for r in ref_list:
                url = r.get("url") if isinstance(r, dict) else getattr(r, "url", None)
                score = (
                    r.get("relevancy_index")
                    if isinstance(r, dict)
                    else getattr(r, "relevancy_index", None)
                )
                if url is None or score is None:
                    continue

                if (url not in best_by_url) or (score > best_by_url[url]["relevancy_index"]):
                    best_by_url[url] = r

            top5 = heapq.nlargest(
                5,
                best_by_url.values(),
                key=lambda x: x["relevancy_index"],
            )

            return {
                "final_answer": aggregated_answer.content + completed_report_answers_non_valid,
                "references": top5,
            }

        # Ingen gyldige del-svar
        return {
            "final_answer": (
                "Jeg beklager, men jeg kan bare svare på spørsmål basert på den gitte konteksten."
            ),
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


def emit_query_answer_references(state: State_Answer) -> Dict[str, Any]:
    """Emitter endelig svar + referanser som events."""
    
    _message( "Aggregating the final answer", event="info")


    q = state.get("query")
    if q:
        _emit(json.dumps(q, ensure_ascii=False), event="Refined query")

    _emit(f"\n## Du spurte\n{state['query']}\n")
    _emit("\n## Svar\n")

    answer = state["final_answer"]
    print(f'Answer:<{answer}>')
    for line in answer.splitlines(True):
        _emit(line, event = "answer")
    _emit("\n", event = "answer")

    top5 = state["references"]

    if top5:
        #_emit("\n## Referanser\n", event = 'references')
        for r in top5:
            bullet = f"- [{r['name']}]({r['url']}) \n"
            _emit(bullet, event = 'references')
            
    usage_payload = json.dumps(
        {
            "input_tokens": state.get("input_tokens", 0),
            "output_tokens": state.get("output_tokens", 0),
        },
        ensure_ascii=False,
    )
    _emit(usage_payload, event="Token usage")
    
    # priser i USD per 1M tokens
    PRICE_INPUT_USD_PER_M = 0.25
    PRICE_OUTPUT_USD_PER_M = 2.0
    USD_TO_NOK = 10  # grovt anslag

    input_tokens = state.get("input_tokens", 0) or 0
    output_tokens = state.get("output_tokens", 0) or 0

    kost_usd = (
        input_tokens * PRICE_INPUT_USD_PER_M
        + output_tokens * PRICE_OUTPUT_USD_PER_M
    ) / 1_000_000

    kost_nok = kost_usd * USD_TO_NOK

    _emit(f"\nKost: {kost_nok:.4f} NOK", event="Token usage")

    return {}


def assign_workers(state: State_Answer) -> List[Send]:
    """Opprett en worker for hver delspørring."""

    _message( "Assign a worker to each section in the plan", event="info")
    
    return [
        Send(
            "query_grounded",
            {
                "subquery": s,
                "query_engine": state["query_engine"],
                "similarity_cutoff": state["similarity_cutoff"],
                "llm": state["llm"],
                "retriever": state["retriever"],
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
        
        print(f'---->query:', state["query"])

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
        _emit(related_queries_payload, event="related queries")
                                           
        
    except Exception as e:
       logging.error(f"Failed to execute agent: {e} ")       
        
    return {"related_queries": related_queries}

# ---------------------------------------------------------
# Bygg workflow
# ---------------------------------------------------------

builder = StateGraph(State_Answer)

builder.add_node("orchestrator", orchestrator)
builder.add_node("query_grounded", query_grounded)
builder.add_node("synthesizer", synthesizer, join=True)
builder.add_node("emit_query_answer_references", emit_query_answer_references)
builder.add_node("related_queries", related_queries)

builder.add_edge(START, "orchestrator")

# Orchestrator → genererer subqueries → assign_workers bestemmer flere query_grounded-noder
builder.add_conditional_edges("orchestrator", assign_workers, ["query_grounded"])

builder.add_edge("query_grounded", "synthesizer")
builder.add_edge("synthesizer", "emit_query_answer_references")
builder.add_edge("emit_query_answer_references", "related_queries")
builder.add_edge("related_queries", END)

answer_workflow = builder.compile()
