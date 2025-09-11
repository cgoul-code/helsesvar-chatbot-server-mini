import os
import re
import logging
import json
from typing import List, Literal
from typing_extensions import TypedDict
from json_utils import safe_parse_json

from llama_index.core.base.response.schema import Response
from llama_index.core.query_engine import BaseQueryEngine

from langgraph.graph import StateGraph, START, END


# === Data types ===
class Reference(TypedDict):
    name: str
    url: str
    relevance_index: float

class RelatedQuery(TypedDict):
    keyword: str
    query: str

class State_AnswerWithRelatedQueries(TypedDict):
    llm: any  # LLM client (from server_settings.get_llm())
    query_engine: BaseQueryEngine
    query_engine_related_queries: BaseQueryEngine
    vector_index_description: str
    query: str
    refinedQuery: str
    similarity_cutoff: float
    relevancy_cutoff: float
    response: Response | None
    validate_response_result: Literal["Accepted", "Rejected"]
    answer: str
    feedback: str
    references: List[Reference]
    related_queries: List[RelatedQuery]
    structured_answer: str


# === Helpers ===



# === Node functions ===
def llm_call_refine_query(state: State_AnswerWithRelatedQueries) -> dict:

    llm = state["llm"]
    query = state["query"]
    msg = llm.invoke(
        f"Please refine the user's question in readable way in norwegian, ensuring that the 'I' form is preserved : {query}e"
    )
    logging.info('---result---: llm_call_refine_query:', msg.content)
    return {"refinedQuery": msg.content}

def llm_call_answer(state: State_AnswerWithRelatedQueries) -> dict:
    
    response_obj = state["query_engine"].query(state["refinedQuery"])
    logging.info('---step---: llm_call_answer:', response_obj.response)
    return {"answer": response_obj.response, "response": response_obj}

def llm_call_related_queries(state: State_AnswerWithRelatedQueries) -> dict:
    # Call the query engine
    response_obj = state["query_engine_related_queries"].query(state["refinedQuery"])
    
    # LlamaIndex Response objects often have the text in .response
    raw = getattr(response_obj, "response", None)
    if raw is None:
        raw = str(response_obj)

    # Try to parse LLM output safely
    try:
        json_obj = safe_parse_json(raw)
    except Exception:
        logging.error("Failed to parse related-queries JSON. Raw:\n%s", raw)
        raise

    # Debugging
    logging.debug("Parsed related queries JSON: %s", json_obj)

    # Convert into your RelatedQuery objects
    relatedQueries: List[RelatedQuery] = []
    for entry in json_obj:
        category = entry.get("Category name", "")
        questions = entry.get("Related questions", [])
        print("Category:", category)
        for q in questions:
            print("  -", q)
            relatedQueries.append(
                {
                    "keyword": category,
                    "query": q,
                }
            )

    return {"related_queries": relatedQueries}
    

def validate_response(state: State_AnswerWithRelatedQueries) -> dict:
    relevancy_cutoff = state["relevancy_cutoff"]
    nodes = [n for n in state["response"].source_nodes if n.score is not None]
    for n in nodes:
        print(f'\nnode score: {n.score } ')
        if n.score >= relevancy_cutoff:
            return {"validate_response_result": "Accepted"}
        
    vector_index_desc = state["vector_index_description"]
    feedback = f'Jeg beklager! {vector_index_desc}. Hvis du har spørsmål om disse emnene, kan jeg prøve å hjelpe deg med det. Bare gi meg beskjed om hva du lurer på!'
    return {"validate_response_result": "Rejected", "feedback": feedback}


def on_reject_build_structured(state: State_AnswerWithRelatedQueries) -> dict:
    print("rejected")
    # exactly what aggregator does:
    return aggregator(state)

def response_builder_node(state: State_AnswerWithRelatedQueries) -> dict:
    return {}


def references_generator(state: State_AnswerWithRelatedQueries) -> dict:
    relevancy_cutoff = state["relevancy_cutoff"]
    refs: List[Reference] = []
    seen = set()  # track already added (e.g. by URL)

    for node in state["response"].source_nodes:
        if node.score is not None and node.score >= relevancy_cutoff:
            meta = node.metadata
            name = meta.get('title', 'Ingen tittel').lstrip()
            url = meta.get('url', 'Ingen URL')
            key = (name, url)  # use tuple to identify uniqueness

            if key not in seen:
                seen.add(key)
                refs.append({
                    "name": name,
                    "url": url,
                    "relevance_index": node.score
                })

    return {"references": refs}


def aggregator(state: State_AnswerWithRelatedQueries) -> dict:
    
    if state["validate_response_result"] == "Rejected":
        feedback = state["feedback"] 
        return {"structured_answer": feedback}
    else:

        combined = f"## Du spurte\n{state['refinedQuery']}\n"
        combined += f"## Svar\n{state['answer']}\n"
        if state['references']:
            combined += "## Referanser\n"
            for r in state['references']:
                #combined += f"- [{r['name']}]({r['url']}) (Relevans: {r['relevance_index']:.2f})\n"
                combined += f"- [{r['name']}]({r['url']}) \n"

        combined += "## Lignende spørsmål\n"
        for r in state["related_queries"]:
                combined += f"=={r['query']}==\n"
                
        return {"structured_answer": combined}


# === Build static, stateless workflow ===
builder = StateGraph(State_AnswerWithRelatedQueries)

# 1️⃣ Core answer + validation
builder.add_node("llm_call_answer", llm_call_answer)
builder.add_node("llm_call_refine_query", llm_call_refine_query)
builder.add_node("validate_response", validate_response)

builder.add_edge(START, "llm_call_refine_query")
builder.add_edge("llm_call_refine_query", "llm_call_answer")
builder.add_edge("llm_call_answer", "validate_response")


# 2️⃣ Branch on validation result:
#    - "Rejected" → aggregator
#    - "Accepted" → fan-out into 4 nodes
builder.add_node("aggregator", aggregator)
builder.add_node("references_generator", references_generator)


builder.add_conditional_edges(
    "validate_response",
    # router: return exactly the keys below
    lambda s: "Rejected" if s["validate_response_result"] == "Rejected"
              else "Accepted",
    {
        "Rejected": "aggregator",
        "Accepted": "references_generator",
    }
)

# 3️⃣ Fan-out the Accepted branch into the other 3 nodes
#    (these edges will only fire when validate_response_result == "Accepted")

#builder.add_edge("validate_response", "references_generator")

builder.add_node("llm_call_related_queries", llm_call_related_queries)
builder.add_edge("references_generator", "llm_call_related_queries")
builder.add_edge("llm_call_related_queries", "aggregator")


# 5️⃣ Finally, aggregator → END
builder.add_edge("aggregator", END)

answer_with_related_queries_workflow = builder.compile()

# produce graph.mmd that visualizes the workflow
#from graph_utils import save_mermaid_diagram
#save_mermaid_diagram(answer_with_related_queries_workflow.get_graph())

logging.info("answer_witth_related_queries_workflow created...")