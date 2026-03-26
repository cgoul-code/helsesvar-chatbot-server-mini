import json
import logging

from operator import add
from typing import Any, Dict, List, Literal, Optional, Tuple

from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from rapidfuzz.fuzz import partial_ratio

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator

from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer
from agent_shared import Reference, _emit, _node_text, _build_related_queries_retriever, _as_int, _as_float, _dedupe_references, _normalize



# ---------------------------------------------------------
# Datamodeller og typer
# ---------------------------------------------------------
    
class State_Related(TypedDict):
    # infra
    llm: Any
    index: VectorStoreIndex # Text-bank index
    index_related_queries: VectorStoreIndex # QA-bank index
 
    categories: List[Dict[str, Any]]

    # query context
    query: str
    conversation_str: str

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
    final_short_answer: str
    
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
    
class RelatedSelection(BaseModel):
    selected_node_ids: List[str] = Field(default_factory=list, max_items=3)
    rationale: str = Field(default="", description="Kort begrunnelse (valgfritt)")
    
class DialogPlan(BaseModel):
    last_user_question: str = Field(description="Siste brukerspørsmål")
    intents: List[NextIntent] = Field(min_items=1, max_items=4)
    
# ---------------------------------------------------------
# Små hjelpefunksjoner
# ---------------------------------------------------------

from typing import Optional, Tuple, List

def _fetch_answer_from_related_question(
    node_id: str,
    index: Optional["VectorStoreIndex"] = None,
    index_qa: Optional["VectorStoreIndex"] = None,
) -> Optional[Tuple[str, str, List["Reference"], str, str]]:
    """
    Fetch QA result by node_id, but ALWAYS read answer + metadata from the QA *Document*
    (ref doc) rather than the node.

    Returns:
        (answer, short_answer, refs, category, severity) or None
    """
    if index_qa is None:
        return None

    ds_qa = index_qa.storage_context.docstore

    # 1) Node is only used to locate the ref document id
    try:
        qa_node = ds_qa.get_node(node_id)
    except Exception as e:
        logging.warning(f"Could not fetch QA node {node_id}: {e}")
        return None

    ref_id = getattr(qa_node, "ref_doc_id", None) or node_id

    # 2) Always read from the QA Document
    try:
        qa_doc = ds_qa.get_document(ref_id)
    except Exception as e:
        logging.warning(f"Could not fetch QA document {ref_id} for node_id={node_id}: {e}")
        return None

    qa_meta = getattr(qa_doc, "metadata", {}) or {}

    answer = (qa_meta.get("answer") or "").strip()
    short_answer = (qa_meta.get("short_answer") or "").strip()

    if not answer:
        # If you want, you can fall back to doc text, but usually answer is in metadata.
        # answer = (getattr(qa_doc, "text", "") or "").strip()
        logging.info(f"No answer found in QA document metadata for ref_id={ref_id}")
        return None

    from_doc_id = (qa_meta.get("from_doc_id") or "").strip()

    # Baseline fields from QA doc (may be overwritten by source doc enrichment)
    title = ""
    url = (qa_meta.get("url") or "").strip()
    icon_url = (qa_meta.get("icon_url") or "").strip()
    category = (qa_meta.get("category") or "").strip()
    severity = (qa_meta.get("severity") or "Green").strip()
    refs = (qa_meta.get("references") or [])

    # 3) Enrich from main/source document (preferred for title/icon/category/severity)
    if index is not None and from_doc_id:
        ds = index.storage_context.docstore
        try:
            src_doc = ds.get_document(from_doc_id)
            src_meta = getattr(src_doc, "metadata", {}) or {}

            title = (src_meta.get("title") or title).strip()
            url = (src_meta.get("url") or url).strip()
            icon_url = (src_meta.get("icon_url") or icon_url).strip()
            category = (src_meta.get("category") or category).strip()
            severity = (src_meta.get("severity") or severity).strip()

        except Exception as e:
            logging.warning(f"Could not fetch source document {from_doc_id}: {e}")


    return answer, short_answer, refs, category, severity



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
            answer, short_answer, refs, category, severity = result
            # IMPORTANT: return the updates; do not mutate `state`
            return {
                "final_answer": answer,
                "final_short_answer": short_answer,
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
        _emit(q, event="Refined query")

    _emit(f"\n## Du spurte\n{state['query']}\n")
    _emit("\n## Svar\n")

    answer = state.get("final_answer", "")
    short_answer = state.get("final_short_answer", "")
    logging.info(f"emit_query_answer_references- Final answer length: {len(answer)} chars; short answer length: {len(short_answer)} chars")
    for line in answer.splitlines(True):
        _emit(line, event = "answer")
    _emit("\n", event = "answer")
    
    for line in short_answer.splitlines(True):
        _emit(line, event = "short_answer")
    _emit("\n", event = "short_answer")

    top5 = state["references"]

    if top5:
        #_emit("\n## Referanser\n", event = 'references')
        for r in top5:
            name = r.get("name", "Uten tittel")
            url = r.get("url", "#")
            icon_url = r.get("icon_url")

            if icon_url:
                bullet = f'[{name}]({url}) ||IMG|| {icon_url}\n'
            else:
                bullet = f'[{name}]({url})\n'
            _emit(bullet, event="references")
   

    return {}

def related_queries(state: State_Related) -> dict:

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
            #main_category=state.get("main_category"),        # may be None → ignored
            main_category=None,        # ignore main_category for broader results
        )

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
        
    return {}

def related_queries_dialog_from_query(state: State_Related) -> dict:
    logging.info(f"related_queries_dialog_from_query- Start processing...")
    try:
        _emit("Related queries: single LLM selection (history-aware)", event="info")

        llm = state["llm"]
        conversation_str = (state.get("conversation_str") or "").strip()
        last_q = (state.get("refined_query") or state.get("query") or "").strip()
        logging.info(f"related_queries_dialog_from_query- Conversation history length: {len(conversation_str)} chars")
        logging.info(f"related_queries_dialog_from_query- Last question: {last_q}")

        # 1) Hent kandidater raskt (uten intents)
        retriever = _build_related_queries_retriever(
            index_qa_bank=state["index_related_queries"],
            top_k=30,          # hent litt bredt, men ikke for mye
            cutoff=0.0,        # la LLM velge
            query_severity=state.get("query_severity"),
            main_category=state.get("main_category"),
        )
        #print(f"related_queries_dialog_from_query- Built retriever for related queries. {state}")

        # Du kan bruke last_q for retrieval (vanligvis best). 
        results = retriever.retrieve(last_q) or []
        #print(f"related_queries_dialog_from_query- Retrieved {len(results)} candidates from retriever.")
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
            f"Samtalehistorikk:\n{conversation_str}\n\n"
            f"Siste spørsmål:\n{last_q}\n\n"
            f"Kandidatspørsmål (JSONL):\n{candidates_jsonl}\n"
        )
        #logging.info(f"--------------------------------\nrelated_queries_dialog_from_query- Invoking LLM for selection...")
        #logging.info(f"related_queries_dialog_from_query- Prompt: {prompt} ")
        #logging.info(f"--------------------------------\nrelated_queries_dialog_from_query- Invoking LLM for selection...")
        selection: RelatedSelection = llm.with_structured_output(RelatedSelection).invoke(prompt)


        selected_ids = selection.selected_node_ids[:2] if selection.selected_node_ids else []
        selected_map = {c["node_id"]: c for c in uniq}

        picked = [selected_map[i] for i in selected_ids if i in selected_map]

        related_queries = [
            {"keyword": p.get("severity", ""), "query": p["text"], "node_id": p["node_id"]}
            for p in picked
        ]

        _emit(json.dumps(related_queries, ensure_ascii=False), event="related queries")
    except Exception as e:
       logging.error(f"Failed to execute agent: {e} ")    
       
    return {}


# ---------------------------------------------------------
# Bygg workflow
# ---------------------------------------------------------

builder = StateGraph(State_Related)

builder.add_node("get_metadata_from_node_id", get_metadata_from_node_id)
builder.add_node("emit_query_answer_references", emit_query_answer_references)
builder.add_node("related_queries_dialog_from_query", related_queries_dialog_from_query)


builder.add_edge(START, "get_metadata_from_node_id")
builder.add_edge("get_metadata_from_node_id", "emit_query_answer_references")
builder.add_edge("emit_query_answer_references", "related_queries_dialog_from_query")
builder.add_edge("related_queries_dialog_from_query", END)

related_qa_workflow = builder.compile()