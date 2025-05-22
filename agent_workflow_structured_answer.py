import os
import re
import logging
from typing import List, Literal
from typing_extensions import TypedDict

from llama_index.core.base.response.schema import Response
from llama_index.core.query_engine import BaseQueryEngine

from langgraph.graph import StateGraph, START, END


# === Data types ===
class Reference(TypedDict):
    name: str
    url: str
    relevance_index: float

class State(TypedDict):
    llm: any  # LLM client (from server_settings.get_llm())
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
    structured_answer: str
    num_iterations: int


# === Helpers ===
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


# === Node functions ===

def llm_call_answer(state: State) -> dict:
    response_obj = state["query_engine"].query(state["query"])
    return {"answer": response_obj.response, "response": response_obj}

def validate_response(state: State) -> dict:
    cutoff = state["similarity_cutoff"]
    nodes = [n for n in state["response"].source_nodes if n.score is not None]
    for n in nodes:
        if n.score >= cutoff:
            return {"validate_response_result": "Accepted"}
        
    vector_index_desc = state["vector_index_description"]
    feedback = f'Jeg beklager! {vector_index_desc}. Hvis du har spørsmål om disse emnene, kan jeg prøve å hjelpe deg med det. Bare gi meg beskjed om hva du lurer på!'
    return {"validate_response_result": "Rejected", "feedback": feedback}

def llm_call_short_version_generator(state: State) -> dict:
    llm = state["llm"]
    query = state["query"]
    msg = llm.invoke(
        f"Give a title in norwegian to the query, ensuring that the 'I' form is preserved: {query}, use only one short sentence"
    )
    return {"query_short_version": msg.content}


def llm_call_summary_generator(state: State) -> dict:
    llm = state["llm"]
    query = state["query"]
    msg = llm.invoke(
        f"Please provide a summary of the user's question in norwegian, ensuring that the 'I' form is preserved : {query}, use only one sentence"
    )
    return {"query_summary": msg.content}

def calculate_readability_index(state: State) -> dict:
    text = state["answer"]
    words = text.split()
    num_words = len(words) or 1
    num_sentences = max(len(re.split(r'[.!?]', text)) - 1, 1)
    num_long = sum(1 for w in words if len(re.sub(r'[^a-zA-Z]', '', w)) > 6)
    lix = (num_words / num_sentences) + (num_long / num_words) * 100

    return {
        "lix_score": lix,
        "lix_category": categorize_lix(lix),
    }



def readability_evaluator(state: State) -> dict:
    # Pull in all three updates from the helper
    updates = calculate_readability_index(state)
    # Optional: log the new values
    print(f'index: {updates["lix_score"]:.2f}, antall {state["num_iterations"]}')

    # Build the base of what we’ll return
    result = {
        # merge in the new lix 
        **updates
    }

    # Decide whether we need another pass
    if (updates["lix_score"] > 35) and (state["num_iterations"] < 4):
        result.update({
            "readable_or_not": "not readable",
            "feedback": "Make this text more readable by using shorter sentences, fewer words, and simpler language."
        })
    else:
        result.update({
            "readable_or_not": "readable",
            "feedback": "No need for improvements"
        })

    return result



def llm_make_answer_more_readable(state: State) -> dict:
    llm = state["llm"]
    answer = state["answer"]
    feedback = state["feedback"]
    msg = llm.invoke(f"Improve readability: {answer}. Feedback: {feedback}")
    new_count = state["num_iterations"] + 1
    return {"answer": msg.content, "num_iterations": new_count}


def route_answer(state: State) -> str:
    return "Accepted" if state["readable_or_not"] == "readable" else "Rejected + Feedback"


def on_reject_build_structured(state: State) -> dict:
    print("rejected")
    # exactly what aggregator does:
    return aggregator(state)

def response_builder_node(state: State) -> dict:
    return {}


def references_generator(state: State) -> dict:
    cutoff = state["similarity_cutoff"]
    refs: List[Reference] = []
    for node in state["response"].source_nodes:
        if node.score is not None and node.score >= cutoff:
            meta = node.metadata
            refs.append({
                "name": meta.get('title', 'Ingen tittel').lstrip(),
                "url": meta.get('url', 'Ingen URL'),
                "relevance_index": node.score
            })
    return {"references": refs}


def aggregator(state: State) -> dict:
    
    if state["validate_response_result"] == "Rejected":
        feedback = state["feedback"] 
        return {"structured_answer": feedback}
    else:
        combined = f"# Oppsummering av spørsmålet\n\n"
        combined += f"## Spørsmålet fra brukeren\n{state['query']}\n\n"
        combined += f"## Tittel\n{state['query_short_version']}\n\n"
        combined += f"## Kort sammendrag av spørsmålet\n{state['query_summary']}\n\n"
        combined += f"## Lettlest svar\n{state['answer']}\n\n"
        if state['references']:
            combined += "## Referanser\n"
            for r in state['references']:
                combined += f"- [{r['name']}]({r['url']}) (Relevans: {r['relevance_index']:.2f})\n"
        return {"structured_answer": combined}


# === Build static, stateless workflow ===
builder = StateGraph(State)

# 1️⃣ Core answer + validation
builder.add_node("llm_call_answer", llm_call_answer)
builder.add_node("validate_response", validate_response)
builder.add_edge(START, "llm_call_answer")
builder.add_edge("llm_call_answer", "validate_response")

# 2️⃣ Branch on validation result:
#    - "Rejected" → aggregator
#    - "Accepted" → fan-out into 4 nodes
builder.add_node("aggregator", aggregator)
builder.add_node("llm_call_short_version_generator", llm_call_short_version_generator)
builder.add_node("llm_call_summary_generator", llm_call_summary_generator)
builder.add_node("references_generator", references_generator)
builder.add_node("readability_evaluator", readability_evaluator)

builder.add_conditional_edges(
    "validate_response",
    # router: return exactly the keys below
    lambda s: "Rejected" if s["validate_response_result"] == "Rejected"
              else "Accepted",
    {
        "Rejected": "aggregator",
        "Accepted": "llm_call_short_version_generator",
    }
)

# 3️⃣ Fan-out the Accepted branch into the other 3 nodes
#    (these edges will only fire when validate_response_result == "Accepted")
builder.add_edge("validate_response", "llm_call_summary_generator")
builder.add_edge("validate_response", "references_generator")
builder.add_edge("validate_response", "readability_evaluator")

builder.add_edge("llm_call_short_version_generator", "aggregator")
builder.add_edge("llm_call_summary_generator", "aggregator")
builder.add_edge("references_generator", "aggregator")

# 4️⃣ Readability loop
builder.add_node("llm_make_answer_more_readable", llm_make_answer_more_readable)
builder.add_conditional_edges(
    "readability_evaluator",
    lambda s: "ok" if s["readable_or_not"] == "readable" else "revise",
    {
        "ok":     "aggregator",
        "revise": "llm_make_answer_more_readable",
    }
)
builder.add_edge("llm_make_answer_more_readable", "readability_evaluator")

# 5️⃣ Finally, aggregator → END
builder.add_edge("aggregator", END)

optimizer_workflow = builder.compile()

# produce graph.mmd that visualizes the workflow
#from graph_utils import save_mermaid_diagram
#save_mermaid_diagram(optimizer_workflow.get_graph())

logging.info("optimizer_workflow created...")