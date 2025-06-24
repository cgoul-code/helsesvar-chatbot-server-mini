import os
import re
import json
import logging
from typing import List, Literal

from llama_index.core.base.response.schema import Response
from llama_index.core.query_engine import BaseQueryEngine
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

# === Data types ===
class Reference(TypedDict):
    name: str
    url: str
    relevance_index: float

class State_StructuredAnswer(TypedDict):
    llm: any
    query_engine: BaseQueryEngine
    vector_index_description: str
    query: str
    similarity_cutoff: float
    response: Response | None
    validate_response_result: Literal["Accepted", "Rejected"]
    answer: str
    lix_score: float
    lix_category: str
    readable_or_not: Literal["readable", "not readable"]
    feedback: str
    references: List[Reference]
    query_short_version: str
    query_summary: str
    final_answer: str
    num_iterations: int


# === Helpers & Node functions ===
def categorize_lix(lix: float) -> str:
    if lix < 25:
        return "Svært lettlest (for barn)"
    if lix < 35:
        return "Lettlest (enkel litteratur, aviser)"
    if lix < 45:
        return "Middels vanskelig (standard aviser, generell sakprosa)"
    if lix < 55:
        return "Vanskelig (akademiske tekster, offisielle dokumenter)"
    return "Svært vanskelig (vitenskapelig litteratur)"


def llm_call_answer(state):
    resp = state["query_engine"].query(state["query"])
    get_stream_writer()({"action": "Calling llm to get the initial answer"})
    return {"answer": resp.response, "response": resp}


def validate_response(state):
    get_stream_writer()({"action": "Validating the response"})
    nodes = [n for n in state["response"].source_nodes if n.score is not None]
    for n in nodes:
        if n.score >= state["similarity_cutoff"]:
            return {"validate_response_result": "Accepted"}
    feedback = (
        f'Jeg beklager! {state["vector_index_description"]}. '
        "Hvis du har spørsmål om disse emnene, kan jeg prøve å hjelpe deg. Gi beskjed!"
    )
    return {"validate_response_result": "Rejected", "feedback": feedback}


def llm_call_short_version_generator(state):
    get_stream_writer()({"action": "Generating a short version"})
    msg = state["llm"].invoke(
        f"Gi en kort tittel på norsk, behold 'I'-formen: {state['query']}"
    )
    return {"query_short_version": msg.content}


def llm_call_summary_generator(state):
    get_stream_writer()({"action": "Generating a summary"})
    msg = state["llm"].invoke(
        f"Oppsummer spørsmålet på norsk i én setning, behold 'I'-formen: {state['query']}"
    )
    return {"query_summary": msg.content}


def references_generator(state):
    get_stream_writer()({"action": "Building references"})
    cutoff = state["similarity_cutoff"]
    refs = []
    for n in state["response"].source_nodes:
        if n.score is not None and n.score >= cutoff:
            m = n.metadata
            refs.append({
                "name": m.get("title", "Ingen tittel").lstrip(),
                "url": m.get("url", "Ingen URL"),
                "relevance_index": n.score
            })
    return {"references": refs}


def calculate_readability_index(state):
    text = state["answer"]
    words = text.split()
    num_words = len(words) or 1
    num_sentences = max(len(re.split(r"[.!?]", text)) - 1, 1)
    num_long = sum(1 for w in words if len(re.sub(r"[^a-zA-Z]", "", w)) > 6)
    lix = (num_words / num_sentences) + (num_long / num_words) * 100
    return {"lix_score": lix, "lix_category": categorize_lix(lix)}


def readability_evaluator(state):
    get_stream_writer()({"action": "Evaluating readability"})
    updates = calculate_readability_index(state)
    if updates["lix_score"] > 35 and state["num_iterations"] < 4:
        return {
            **updates,
            "readable_or_not": "not readable",
            "feedback": "Make this text more readable: shorter sentences & simpler language."
        }
    return {
        **updates,
        "readable_or_not": "readable",
        "feedback": "No need for improvements"
    }


def llm_make_answer_more_readable(state):
    get_stream_writer()({"action": "Making answer more readable"})
    msg = state["llm"].invoke(
        f"Improve readability: {state['answer']}. Feedback: {state['feedback']}"
    )
    return {"answer": msg.content, "num_iterations": state["num_iterations"] + 1}


def aggregator(state):
    get_stream_writer()({"action": "Aggregating final response"})
    if state["validate_response_result"] == "Rejected":
        return {"final_answer": state["feedback"]}
    parts = [
        "# Oppsummering av spørsmålet\n\n",
        "## Spørsmålet\n", state["query"], "\n\n",
        "## Tittel\n", state["query_short_version"], "\n\n",
        "## Sammendrag\n", state["query_summary"], "\n\n",
        "## Svar\n", state["answer"], "\n\n"
    ]
    if state["references"]:
        parts.append("## Referanser\n")
        for r in state["references"]:
            parts.append(
                f"- [{r['name']}]({r['url']}) (Relevans: {r['relevance_index']:.2f})\n"
            )
    get_stream_writer()({"final_answer": "".join(parts)})
    return {"final_answer": "".join(parts)}


def collect_all_metadata(state):
    # no-op sink for fan-out
    return {}


# === Build the StateGraph ===
builder = StateGraph(State_StructuredAnswer)

# 1) Answer + validation
builder.add_node("llm_call_answer", llm_call_answer)
builder.add_node("validate_response", validate_response)
builder.add_edge(START, "llm_call_answer")
builder.add_edge("llm_call_answer", "validate_response")

# 2) Rejection shortcut
builder.add_node("aggregator", aggregator)
builder.add_conditional_edges(
    "validate_response",
    lambda s: "Rejected" if s["validate_response_result"]=="Rejected" else "Accepted",
    {"Rejected": "aggregator", "Accepted": "collect_all_metadata"}
)

# 3) Metadata fan-out
builder.add_node("collect_all_metadata", collect_all_metadata)
builder.add_node("llm_call_short_version_generator", llm_call_short_version_generator)
builder.add_node("llm_call_summary_generator", llm_call_summary_generator)
builder.add_node("references_generator", references_generator)
builder.add_edge("collect_all_metadata", "llm_call_short_version_generator")
builder.add_edge("collect_all_metadata", "llm_call_summary_generator")
builder.add_edge("collect_all_metadata", "references_generator")

# 4) Join metadata → metadata_ready
builder.add_node("metadata_ready", lambda s: {})
builder.add_edge("llm_call_short_version_generator", "metadata_ready")
builder.add_edge("llm_call_summary_generator", "metadata_ready")
builder.add_edge("references_generator", "metadata_ready")

# 5) Readability loop
builder.add_node("readability_evaluator", readability_evaluator)
builder.add_node("llm_make_answer_more_readable", llm_make_answer_more_readable)
builder.add_edge("metadata_ready", "readability_evaluator")
builder.add_conditional_edges(
    "readability_evaluator",
    lambda s: "ok"     if s["readable_or_not"]=="readable" else "revise",
    {"ok": "aggregator", "revise": "llm_make_answer_more_readable"}
)
builder.add_edge("llm_make_answer_more_readable", "readability_evaluator")

# 6) Finish
builder.add_edge("aggregator", END)

optimizer_workflow = builder.compile()

# produce graph.mmd that visualizes the workflow
# from graph_utils import save_mermaid_diagram
# save_mermaid_diagram(optimizer_workflow.get_graph())

logging.info("optimizer_workflow created...")