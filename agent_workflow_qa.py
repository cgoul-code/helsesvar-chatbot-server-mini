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

class Reference(TypedDict):
    name: str
    url: str
    icon_url: str
    relevancy_index: float
    
    
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
    print(f'<<<{prompt}>>>')
    return llm.with_structured_output(DialogPlan).invoke(prompt)

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
    qa_node = ds_qa.get_node(node_id)
    
    meta = getattr(qa_node, "metadata", {}) or {}
    answer = meta.get("answer", "")
    short_answer = meta.get("short_answer","")
    from_doc_id = meta.get("from_doc_id", "")
    
    if not answer:
        return None
    
    title = ""
    url = meta.get("url", "")
    icon_url = ""
    category = meta.get("category", "")
    severity = meta.get("severity", "Green")
    
    if index is not None and from_doc_id:
        ds = index.storage_context.docstore
        try:
            src_doc = ds.get_document(from_doc_id)  # <-- THIS is the key
            text_meta = getattr(src_doc, "metadata", {}) or {}
            title = text_meta.get("title", title)
            url = text_meta.get("url", url)
            icon_url = text_meta.get("icon_url", icon_url)
            category = text_meta.get("category", category)
            severity = text_meta.get("severity", severity)
        except Exception as e:
            logging.warning(f"Could not fetch document {from_doc_id}: {e}")


    refs: List[Reference] = [
        {
            "name": title or url or "Uten tittel",
            "url": url,
            "icon_url": icon_url,
            "relevancy_index": 0.0,
        }
    ]

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

    for line in answer.splitlines(True):
        _emit(line, event = "answer")
    _emit("\n", event = "answer")
    
    for line in short_answer.splitlines(True):
        _emit(line, event = "short_answer")
        print(f'###short:{line}')
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
            main_category=state.get("main_category"),        # may be None → ignored
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

# def related_queries_dialog_from_query(state: State_Related) -> dict:
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
#     return {}

def related_queries_dialog_from_query(state: State_Related) -> dict:
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
        "- Velg MAKS 3 kandidatspørsmål som er en NATURLIG fortsettelse i dialogen.\n"
        "- Ikke velg kandidater som bare gjentar siste spørsmål.\n"
        "- Velg kandidater som flytter dialogen videre (avklaring, neste steg, risiko, når søke hjelp).\n"
        "- IKKE endre teksten i kandidatene. Du skal bare returnere node_id.\n"
        "- Hvis ingen passer, returner tom liste.\n\n"
        f"Samtalehistorikk:\n{conversation_history}\n\n"
        f"Siste spørsmål:\n{last_q}\n\n"
        f"Kandidatspørsmål (JSONL):\n{candidates_jsonl}\n"
    )

    selection: RelatedSelection = llm.with_structured_output(RelatedSelection).invoke(prompt)

    selected_ids = selection.selected_node_ids[:3] if selection.selected_node_ids else []
    selected_map = {c["node_id"]: c for c in uniq}

    picked = [selected_map[i] for i in selected_ids if i in selected_map]

    related_queries = [
        {"keyword": p.get("severity", ""), "query": p["text"], "node_id": p["node_id"]}
        for p in picked
    ]

    _emit(json.dumps(related_queries, ensure_ascii=False), event="related queries")
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