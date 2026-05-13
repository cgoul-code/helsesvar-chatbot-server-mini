from llama_index_client import FilterOperator
from agent_workflow_answer import (answer_workflow, State_Answer)
from agent_workflow_qa import (related_qa_workflow, State_Related)
from config import ServerSettings, VectorIndexStore, CustomError
from query_utils import QuerySettings
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.core.query_engine import RetrieverQueryEngine 
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate, get_response_synthesizer, VectorStoreIndex
from typing import Literal
import logging
import asyncio
import re
import random
from typing import Any, Dict, List, Optional, Tuple
from llama_index.core.schema import NodeWithScore
from llama_index.core.postprocessor import SimilarityPostprocessor


# Fixed fallback used when no index named "{vectorIndex}_qa_bank" exists.
# The QA-bank is only used for the related-queries suggestions feature; it is
# not load-bearing for answer retrieval. So we can decouple it from the
# answer-retrieval index name (e.g. "hvaerinnafor_unified") and reuse the
# original question bank for follow-up suggestions regardless.
QA_BANK_FALLBACK_NAME = "hvaerinnafor_qa_bank"


def _resolve_qa_bank_entry(query_settings, vector_store):
    """Pick which loaded index to use as the QA-bank.

    Resolution order:
      1. If the request supplies `qa_bank_index` and that index is loaded, use it.
      2. Otherwise try `{vectorIndex}_qa_bank` (legacy convention).
      3. Otherwise fall back to QA_BANK_FALLBACK_NAME so the answer-retrieval
         index can be swapped independently.

    Returns (resolved_name, entry_or_None).
    """
    requested = getattr(query_settings, "qa_bank_index", None)
    if requested:
        entry = vector_store.get(requested)
        if entry is not None:
            return requested, entry
        logging.warning(
            "Requested qa_bank_index %s not loaded; falling back to default resolution.",
            requested,
        )

    name = f"{query_settings.vectorIndex}_qa_bank"
    entry = vector_store.get(name)
    if entry is not None:
        return name, entry

    if QA_BANK_FALLBACK_NAME != name:
        fallback = vector_store.get(QA_BANK_FALLBACK_NAME)
        if fallback is not None:
            logging.warning(
                "QA-bank %s not found; using %s as fallback for related-queries.",
                name, QA_BANK_FALLBACK_NAME,
            )
            return QA_BANK_FALLBACK_NAME, fallback

    return name, None


categories = [
  {
    "name": "Eksen",
    "subcategories": ["eks","eksen","eks-kjæreste","ex","skal jeg gå tilbake til eksen","følelser for eksen","forelsket i eksen","hvordan få eksen tilbake","vinne tilbake eksen","jeg savner eksen min","sammen med eksen igjen","tenker på eksen","eksen savner meg","når eksen angrer","eksen vil ha meg tilbake","venn med eksen","hvordan komme over eksen"],
    "weight": 17,
    "related": [
      {"name":"Kjæreste / Forhold","weight":27},
      {"name":"Vennskap","weight":13},
      {"name":"Kjærlighetssorg","weight":9}
    ]
  },
  {
    "name": "Kjæreste / Forhold",
    "subcategories": ["kjæreste","kjærester","par","partner","parforhold","forhold","sammen","holde på","problemer","krangler","sjalu","sjalusi","usikkerhet","tillitsbrudd","kommunikasjon","forventning","samtale","praten","avstandsforhold","vold","overgrep","sint","mistenksom","svik","tilgi","skam","skyldfølelse"],
    "weight": 27,
    "related": [
      {"name":"Eksen","weight":17},
      {"name":"Vennskap","weight":13},
      {"name":"Ekteskap","weight":7},
      {"name":"Utroskap","weight":7},
      {"name":"Kjærlighetssorg","weight":9},
      {"name":"Sex / Intimitet","weight":24},
      {"name":"Forelskelse / Flørting","weight":29}
    ]
  },
  {
    "name": "Sex / Intimitet",
    "subcategories": ["sex","samleie","kåt","lyst på sex","lite lyst","liten lyst","frustrert","frustrerende","sex med en venn","vennesex","sex med en kompis","sex med en kollega","må man være kjærester for å ha sex?","hvem kan jeg ha sex med?","overtale noen til sex","overnatting","overnatte","sove sammen","ligge sammen","sjekke opp","hooke","første kyss","tungekyss","kyssing"],
    "weight": 24,
    "related": [
      {"name":"Kjæreste / Forhold","weight":27},
      {"name":"Utroskap","weight":7},
      {"name":"Ekteskap","weight":7},
      {"name":"Forelskelse / Flørting","weight":29}
    ]
  },
  {
    "name": "Forelskelse / Flørting",
    "subcategories": ["forelskelse","forelsket","forelska","betatt","elsker","besatt","oppslukt","kjendisforelskelse","justin bieber","første forelskelse","test forelsket","tegn på forelskelse","fysiske symptomer","hvordan få noen til å like deg","hvordan få kjæreste","hvordan få henne/han interessert","liker han meg","liker hun meg","flørt","flørting","flørteskole","flørtetips","blikkontakt","initiativ","første steg","signaler","snapchat-flørting","meldinger","bilder"],
    "weight": 29,
    "related": [
      {"name":"Sex / Intimitet","weight":24},
      {"name":"Kjærlighetssorg","weight":9},
      {"name":"Ungdom / Sosiale tema","weight":19},
      {"name":"Aktiviteter / Tips","weight":11}
    ]
  },
  {
    "name": "Vennskap",
    "subcategories": ["venn","venner","venninner","kamerater","bestevenn","vanskelig vennskap","dårlige venner","avslutte vennskap","avsluttet vennskap","si ifra til en venn","vennen min hører ikke på meg","vennen min er forelsket i meg","forelska i en venn"],
    "weight": 13,
    "related": [
      {"name":"Eksen","weight":17},
      {"name":"Aktiviteter / Tips","weight":11},
      {"name":"Ungdom / Sosiale tema","weight":19},
      {"name":"Kjæreste / Forhold","weight":27}
    ]
  },
  {
    "name": "Ekteskap",
    "subcategories": ["ekteskap","ekteskapsloven","gift","vie","bryllup","kirkebryllup","borgerlig vielse"],
    "weight": 7,
    "related": [
      {"name":"Kjæreste / Forhold","weight":27},
      {"name":"Sex / Intimitet","weight":24}
    ]
  },
  {
    "name": "Utroskap",
    "subcategories": ["utro","utroskap","usikkerhet","tillitsbrudd","svik","skam","skyldfølelse"],
    "weight": 7,
    "related": [
      {"name":"Sex / Intimitet","weight":24},
      {"name":"Kjæreste / Forhold","weight":27}
    ]
  },
  {
    "name": "Kjærlighetssorg",
    "subcategories": ["kjærlighetssorg","hvordan komme seg over kjærlighetssorg","hjelp mot kjærlighetssorg","behandling av kjærlighetssorg","tips mot kjærlighetssorg","råd mot kjærlighetssorg","redd for å bli avvist","avvisning","avvist"],
    "weight": 9,
    "related": [
      {"name":"Forelskelse / Flørting","weight":29},
      {"name":"Eksen","weight":17},
      {"name":"Kjæreste / Forhold","weight":27}
    ]
  },
  {
    "name": "Ungdom / Sosiale tema",
    "subcategories": ["muslim","muslimsk kjæreste","strenge foreldre","kultur","barneloven","kontroll","foreldre","nei","regler","alder","aldersforskjell (ung/gammel/eldre)","fest","ferie","tur","syden","alkohol","drikker","fantasi","avtaler"],
    "weight": 19,
    "related": [
      {"name":"Vennskap","weight":13},
      {"name":"Aktiviteter / Tips","weight":11},
      {"name":"Forelskelse / Flørting","weight":29}
    ]
  },
  {
    "name": "Aktiviteter / Tips",
    "subcategories": ["daten","kino","middag","aktivitet","finne på","crush","tips","råd","hva skal vi gjøre","snakke","småprat"],
    "weight": 11,
    "related": [
      {"name":"Ungdom / Sosiale tema","weight":19},
      {"name":"Vennskap","weight":13},
      {"name":"Forelskelse / Flørting","weight":29}
    ]
  }
]
async def get_answer_as_stream(
    query_settings: QuerySettings,
    server_settings: ServerSettings,
    vector_store: VectorIndexStore
):
  try:
    # Build a conversation string from messages (if any)
    history = query_settings.messages or []
    convo_lines = []
    for m in history:
        role = m.get("role", "user")
        content = m.get("content", "")
        if not content:
            continue
        if role == "user":
            convo_lines.append(f"Bruker: {content}")
        elif role == "assistant":
            convo_lines.append(f"Veileder: {content}")
        else:
            convo_lines.append(f"{role}: {content}")

    conversation_str = "\n".join(convo_lines)

    # Extract last user question (already done in QuerySettings)
    last_question = query_settings.user_content or query_settings.query
  
    #   # Combine for the workflow:
    # if conversation_str:
    #     full_query = (
    #         f"Tidligere samtale mellom bruker og veileder:\n"
    #         f"{conversation_str}\n\n"
    #         f"Siste spørsmål fra bruker som du skal svare på nå:\n"
    #         f"{last_question}"
    #     )
    # else:
    #     full_query = last_question
  
    # 1) Try to load the requested index
    vec_name = query_settings.vectorIndex
    entry = vector_store.get(vec_name)
    if entry is None:
        # Log with %s formatting
        logging.error("Index not found: %s", vec_name)
        # Raise so the route handler can catch & return 404 JSON
        raise CustomError(
            f"Index not found, referansefilene for {vec_name} mangler!",
            404
        )

    index: VectorStoreIndex = entry.index
    vector_index_description = entry.description
    logging.info("Found entry: %s", vector_index_description)
    
    # 1.1 Load the qa_bank corresponding to the index
    vec_name_qa_bank, entry_qa_bank = _resolve_qa_bank_entry(query_settings, vector_store)
    if entry_qa_bank is None:
        logging.error("Index not found: %s (and no fallback)", vec_name_qa_bank)
        raise CustomError(
            f"Index not found, referansefilene for {vec_name_qa_bank} mangler!",
            404
        )

    index_qa_bank: VectorStoreIndex = entry_qa_bank.index

    # 2) Build your prompt template
    text_qa_template = ChatPromptTemplate([

      ChatMessage(
          role=MessageRole.SYSTEM,
          content=(
              "Du er en hjelpsom og nøyaktig veileder som svarer på norsk (bokmål).\n\n"
              "VIKTIGE REGLER:\n"
              "- Du skal bruke SAMTALEHISTORIKKEN kun for å forstå sammenhengen og hva brukeren mener.\n"
              "- Du skal bruke KUN 'konteksten' nedenfor som faktagrunnlag når du svarer.\n"
              "- Hvis svaret ikke finnes i konteksten, svar nøyaktig: \"Det vet jeg ikke basert på kildene.\"\n"
              "- Hver faktapåstand i svaret må ha minst én kilde i 'citations'.\n"
              "- Ikke legg til, anta eller forklare noe som ikke står eksplisitt i konteksten.\n"
          )
      ),
      ChatMessage(
          role=MessageRole.USER,
          content=(
              "SAMTALEHISTORIKK (kun for kontekst, ikke som faktagrunnlag):\n"
              "---------------------\n"
              "{conversation_str}\n"
              "---------------------\n\n"
              "KONTEKSTINFORMASJON (dette er eneste faktagrunnlag):\n"
              "---------------------\n"
              "{context_str}\n"
              "---------------------\n\n"
              "Spørsmål:\n"
              "{query_str}\n\n"
              "Svar:"
          ),
      ),
    ])


    # 3) Create the response synthesizer
    response_synthesizer = get_response_synthesizer(
        response_mode=query_settings.response_mode,
        text_qa_template=text_qa_template,
        summary_template=text_qa_template,
        structured_answer_filtering=True,
        streaming=False,
        verbose=True,
    )
    
    # 4) Configure the query engine
    query_engine = index.as_query_engine(
        similarity_cutoff=query_settings.similarity_cutoff,
        similarity_top_k=query_settings.similarity_top_k,
        response_synthesizer=response_synthesizer,
    )
    retriever = index.as_retriever(
        similarity_top_k=query_settings.similarity_top_k,
        similarity_cutoff=query_settings.similarity_cutoff,
    )
    
     # 6) define a related question template
    text_rq_template = ChatPromptTemplate([
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(                
                "You will receive a user query and a list of CANDIDATE QUERIES, each with a unique 'id' and a 'text'.\n"
                "Your task is to select the most relevant candidate queries to the user query. IMPORTANT RULES:\n"
                "1) DO NOT REWRITE OR EDIT ANY CANDIDATE TEXT.\n"
                "2) Output ONLY the IDs of the selected candidates in JSON.\n"
                "3) Prefer candidates that best capture the user's information need; avoid near-duplicates.\n"
                "4) If nothing is clearly relevant, return an empty list.\n\n"
                "USER QUERY:\n{user_query}\n\n"
                "CANDIDATE QUERIES (JSON Lines, one per line):\n{candidates_jsonl}\n\n"
                "Output JSON ONLY (no markdown):\n"
                '{{"selected_ids": ["id1","id2","id3"]}}'
            )
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=(
                "Context information is below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Given the context information and not prior knowledge, "
                "Query: {query_str}\n"
                "Answer: "
            ),
        ),
    ])
    severity_filters = MetadataFilters(
        filters=[MetadataFilter(key="severity", value=v) for v in ("Green", "Yellow", "Red")], condition="or",
    )
    
    retriever_related_queries = index_qa_bank.as_retriever(
        similarity_top_k=query_settings.similarity_top_k,
        similarity_cutoff=query_settings.similarity_cutoff,
        filters=severity_filters,
    )

    # Tier-1 retrieval: real Q&A from ung.no (psa_ssa_topics).
    # Loaded if available; the cascade in query_grounded skips it if missing
    # or if the user is querying psa_ssa_topics directly.
    index_psa_ssa = None
    retriever_psa_ssa = None
    if query_settings.vectorIndex != "psa_ssa_topics":
        entry_psa_ssa = vector_store.get("psa_ssa_topics")
        if entry_psa_ssa is not None:
            index_psa_ssa = entry_psa_ssa.index
            retriever_psa_ssa = index_psa_ssa.as_retriever(
                similarity_top_k=5,
                # No similarity_cutoff here — we apply psa_ssa_threshold ourselves
                # in the cascade so we can decide based on the top score.
            )
        else:
            logging.info("psa_ssa_topics index not loaded; tier-1 cascade disabled")
         
    response_related_queries_synthesizer = get_response_synthesizer(
        response_mode= "tree_summarize",
        summary_template= text_rq_template, 
        streaming = True,
        structured_answer_filtering=True, 
        verbose=True)
    
    from llama_index.core.postprocessor import SimilarityPostprocessor

    query_engine_related_queries = index.as_query_engine(
        similarity_top_k=query_settings.similarity_top_k,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=query_settings.similarity_cutoff)],
        response_synthesizer=response_related_queries_synthesizer,
    )
    # query_engine_related_queries = RetrieverQueryEngine.from_args(
    #     node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=query_settings.similarity_cutoff)],
    #     retriever=retriever_related_queries,
    #     response_synthesizer=response_related_queries_synthesizer,
    # )


    # 5) Initialize and run your optimizer workflow
    
    init_state: State_Answer = {
        "llm": server_settings.llm,
        "index" : index,
        "query_engine": query_engine,
        "retriever": retriever,
        "vector_index_description": vector_index_description,
        #"query": query_settings.user_content,
        "query": last_question,
        "conversation_str": conversation_str,
        "index_related_queries" :index_qa_bank,
        "retriever_related_queries" : retriever_related_queries,
        "index_psa_ssa": index_psa_ssa,
        "retriever_psa_ssa": retriever_psa_ssa,

        "from_node_id": query_settings.from_node_id,
        "similarity_cutoff": query_settings.similarity_cutoff,
        "similarity_top_k": query_settings.similarity_top_k,
        "relevancy_cutoff" : query_settings.relevancy_cutoff,
        "psa_ssa_threshold": query_settings.psa_ssa_threshold,

        # defaults:
        "relevancy_band": "",
        "best_node_score": 0.0,
        "response": None,
        "validate_response_result": "Rejected",
        "answer": "",
        "feedback": "",
        "references": [],
        "subqueries": [],
        "completed_subqueries":[],
        "final_answer": "",
        "final_short_answer": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "refined_query": "",
        "needs_subqueries": False,
        "main_category": "",
        "query_severity": "",
        "stance": "",
        "harm_to_others_tense": "na",
        "response_style": query_settings.response_style,
    }




    # ✅ This is  an **async generator**
    async for chunk in answer_workflow.astream(init_state, stream_mode="custom"):
        yield chunk
  except CustomError:
      # Forventede feil – la route håndtere HTTP-respons
      logging.warning("CustomError raised in get_answer_as_stream", exc_info=True)
      raise

  except asyncio.CancelledError:
      # Viktig: må ikke svelges i async streaming
      logging.info("Streaming cancelled by client")
      raise

  except Exception as e:
      # Uventede feil
      logging.exception("Unhandled error in get_answer_as_stream")
      raise CustomError(
          "En intern feil oppstod under behandling av forespørselen.",
          500
      )
# --------------------------------------------------------------------------------------------------------------

async def get_related_qa_as_stream(
    query_settings: QuerySettings,
    server_settings: ServerSettings,
    vector_store: VectorIndexStore
):
  try:
    # 1) Load the text_bank corresponding to the index
    vec_name = query_settings.vectorIndex
    entry = vector_store.get(vec_name)
    if entry is None:
        # Log with %s formatting
        logging.error("Index not found: %s", vec_name)
        # Raise so the route handler can catch & return 404 JSON
        raise CustomError(
            f"Index not found, referansefilene for {vec_name} mangler!",
            404
        )

    index: VectorStoreIndex = entry.index

    # 1) Load the qa_bank corresponding to the index
    vec_name_qa_bank, entry_qa_bank = _resolve_qa_bank_entry(query_settings, vector_store)
    if entry_qa_bank is None:
        logging.error("Index not found: %s (and no fallback)", vec_name_qa_bank)
        raise CustomError(
            f"Index not found, referansefilene for {vec_name_qa_bank} mangler!",
            404
        )
    index_qa_bank: VectorStoreIndex = entry_qa_bank.index
    
    history = query_settings.messages or []
    convo_lines = []
    for m in history:
        role = m.get("role", "user")
        content = m.get("content", "")
        if not content:
            continue
        if role == "user":
            convo_lines.append(f"Bruker: {content}")
        elif role == "assistant":
            convo_lines.append(f"Veileder: {content}")
        else:
            convo_lines.append(f"{role}: {content}")

    conversation_str = "\n".join(convo_lines)
    logging.info(f'<conversation_str: {conversation_str}>')

    # Extract last user question (already done in QuerySettings)
    last_question = query_settings.user_content or query_settings.query
  
    print(f'query_settings.from_node_id: {query_settings.from_node_id}')

    # 2) Initialize and run your optimizer workflow
    init_state: State_Related = {
        "llm": server_settings.llm,
        "index": index,
        "index_related_queries" : index_qa_bank,
        "categories": categories,
        "query": last_question,
        "conversation_str" : conversation_str,
        "from_node_id": query_settings.from_node_id,
        
        "similarity_cutoff": query_settings.similarity_cutoff,
        "similarity_top_k": query_settings.similarity_top_k,
        "relevancy_cutoff" : query_settings.relevancy_cutoff,
        
        # defaults:
        "main_category": "",
        "query_severity": "",
        "references": [],
        "final_answer": "",
        "final_short_answer": ""
    }



    # ✅ This is  an **async generator**
    async for chunk in related_qa_workflow.astream(init_state, stream_mode="custom"):
      yield chunk
        
  except CustomError:
    # Forventede feil – la route håndtere HTTP-respons
    logging.warning("CustomError raised in get_answer_as_stream", exc_info=True)
    raise

  except asyncio.CancelledError:
    # Viktig: må ikke svelges i async streaming
    logging.info("Streaming cancelled by client")
    raise

  except Exception as e:
    # Uventede feil
    logging.exception("Unhandled error in get_answer_as_stream")
    raise CustomError(
        "En intern feil oppstod under behandling av forespørselen.",
        500
    )

# --------------------------------------------------------------------------------------------------------------
def _category_title_to_id(title: str) -> str:
    s = (title or "").strip().lower()
    s = re.sub(r"\s+", "-", s)
    # keep norwegian letters too
    s = re.sub(r"[^a-z0-9\-æøå]", "", s)
    return s or "category"

def _node_to_query_and_id(nws: Any) -> Optional[Dict[str, Any]]:
    """
    Convert a LlamaIndex NodeWithScore (or similar) to {query, node_id}.
    Tries metadata first (question/query/title), then falls back to node text.
    """
    try:
        node = nws.node if hasattr(nws, "node") else nws
        meta = getattr(node, "metadata", None) or {}

        q = (
            meta.get("question")
            or meta.get("query")
            or meta.get("title")
            or meta.get("q")
            or None
        )

        if not q:
            if hasattr(node, "get_content"):
                q = node.get_content(metadata_mode="none")
            elif hasattr(node, "get_text"):
                q = node.get_text()
            else:
                q = str(node)

        q = (q or "").strip()
        if not q:
            return None

        node_id = getattr(node, "node_id", None) or meta.get("node_id")
        return {"query": q, "node_id": node_id}
    except Exception:
        return None

def _pick_unique_random(items: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    if not items:
        return []
    # dedupe by query (case-insensitive)
    seen = set()
    deduped = []
    for it in items:
        q = (it.get("query") or "").strip().lower()
        if not q or q in seen:
            continue
        seen.add(q)
        deduped.append(it)

    if len(deduped) <= k:
        random.shuffle(deduped)
        return deduped
    return random.sample(deduped, k)

def _split_joined_category(value: str) -> List[str]:
    """
    Supports:
      "A"     -> ["A"]
      "A|B"   -> ["A", "B"]
      "|A|B|" -> ["A", "B"]
    """
    if not isinstance(value, str):
        return []
    s = value.strip()
    if not s:
        return []
    return [p.strip() for p in s.split("|") if p.strip()]
async def _discover_categories_from_qabank(index_qa_bank: VectorStoreIndex) -> List[str]:
    cats = set()

    sc = getattr(index_qa_bank, "storage_context", None)
    ds = getattr(sc, "docstore", None) if sc else None
    docs_dict = getattr(ds, "docs", None) if ds else None

    if not isinstance(docs_dict, dict):
        return []

    for doc in docs_dict.values():
        meta = getattr(doc, "metadata", None) or {}

        raw_list = meta.get("categories")
        if isinstance(raw_list, list):
            for x in raw_list:
                s = str(x).strip()
                if s:
                    cats.add(s)

        raw = meta.get("category")
        if isinstance(raw, str) and raw.strip():
            for part in [p.strip() for p in raw.split("|") if p.strip()]:
                cats.add(part)

        raw2 = meta.get("main_category")
        if isinstance(raw2, str) and raw2.strip():
            cats.add(raw2.strip())

    return sorted(cats)


def _meta_has_category(meta: dict, cat_title: str) -> bool:
    cat_title = (cat_title or "").strip()
    if not cat_title:
        return False

    # categories: list of up to 3 categories
    cats = meta.get("categories")
    if isinstance(cats, list) and any(str(x).strip() == cat_title for x in cats):
        return True

    # category: hovedkategori (single string)
    hoved = meta.get("category")
    if isinstance(hoved, str) and hoved.strip() == cat_title:
        return True

    return False

def _meta_is_valid(meta: dict) -> bool:
    v = (meta or {}).get("valid", True)  # default True if missing (change if you prefer)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v == 1
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return False

# ------------------------------
# NEW AGENT: /examples (one response, full list)
# ------------------------------
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator

async def get_examples_full_as_stream(
    query_settings: QuerySettings,
    server_settings: ServerSettings,
    vector_store: VectorIndexStore,
):
    logging.info("------------->>get_examples_full_as_stream")

    try:
        vec_name_qa_bank, entry_qa_bank = _resolve_qa_bank_entry(query_settings, vector_store)
        if entry_qa_bank is None:
            logging.error("Index not found: %s (and no fallback)", vec_name_qa_bank)
            raise CustomError(
                f"Index not found, referansefilene for {vec_name_qa_bank} mangler!",
                404
            )

        index_qa_bank: VectorStoreIndex = entry_qa_bank.index

        requested = getattr(query_settings, "requested_categories", None)
        if isinstance(requested, list) and requested:
            categories_to_use = [str(x).strip() for x in requested if str(x).strip()]
        else:
            categories_to_use = await _discover_categories_from_qabank(index_qa_bank)

        if not categories_to_use:
            yield {"event": "examples_categories", "items": []}
            yield {"event": "done"}
            return

        out: List[Dict[str, Any]] = []

        # Optional: try to pre-filter by metadata.valid == True in the retriever
        # (Some vector stores support this, others ignore/raise — we handle safely.)
        valid_filters = MetadataFilters(
            filters=[MetadataFilter(key="valid", value=1, operator=FilterOperator.EQ)]
        )

        for cat_title in categories_to_use:
            cat_title = (cat_title or "").strip()
            if not cat_title:
                continue

            mapped: List[Dict[str, Any]] = []

            # --- Retrieve a big pool ---
            try:
                try:
                    # Try retriever-level filter (best if supported)
                    retriever = index_qa_bank.as_retriever(
                        similarity_top_k=200,
                        similarity_cutoff=0.0,
                        filters=valid_filters,
                    )
                except TypeError:
                    # Older versions / stores that don't accept filters here
                    retriever = index_qa_bank.as_retriever(
                        similarity_top_k=200,
                        similarity_cutoff=0.0,
                    )

                results = await retriever.aretrieve(cat_title)

            except Exception:
                logging.exception("Retriever failed for category=%r", cat_title)
                results = []

            # --- Post-filter strictly (always) ---
            for nws in results or []:
                node = getattr(nws, "node", None)
                if node is None:
                    continue

                item = _node_to_query_and_id(nws)
                if item:
                    mapped.append(item)

            picked = _pick_unique_random(mapped, 3)

            out.append({
                "id": _category_title_to_id(cat_title),
                "title": cat_title,
                "items": picked,
            })

        yield {"event": "examples_categories", "items": out}
        yield {"event": "done"}

    except CustomError:
        raise
    except asyncio.CancelledError:
        raise
    except Exception:
        logging.exception("Unhandled error in get_examples_full_as_stream")
        raise CustomError("En intern feil oppstod under behandling av forespørselen.", 500)