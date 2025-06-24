import os
import re
import logging
from typing import List, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from llama_index.core.base.response.schema import Response
from llama_index.core.query_engine import BaseQueryEngine


from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.constants import Send

class Reference(TypedDict):
    name: str
    url: str
    relevancy_index: float   

# Schema for structured output to use in planning
class SubQuery(BaseModel):
    subquery: str = Field(
        description="The subquery",
    )
    answer: str = Field(
        description="Answer to the subquery",
    )
    references : List[Reference] = Field(
        description="list of references",
    )
    response_validity: Literal["valid", "not valid"]


class SubQueries(BaseModel):
    subqueries: List[SubQuery] = Field(
        description="Sections of the structured answer.",
    )

# Graph state
class State_SubqueryOrchestrator(TypedDict):
    llm: any  # LLM client (from server_settings.get_llm())
    query_engine: BaseQueryEngine
    similarity_cutoff: float
    query: str  # Report topic
    subqueries: list[SubQuery]  # List of subqueries
    final_answer: str  # Final report
    
# Worker state
class WorkerState(TypedDict):
    subquery: SubQuery
    similarity_cutoff: float
    query_engine: BaseQueryEngine
    
# Nodes
def orchestrator(state: State_SubqueryOrchestrator):
    """Orchestrator that generates a plan for solving the question"""
    writer = get_stream_writer()  
    writer({"action": "Orchestrator that generates a plan for solving the question"})
    
    # Generate queries
    llm = state["llm"]
        
    # Augment the LLM with schema for structured output
    planner = llm.with_structured_output(SubQueries)

    report_queries = planner.invoke(
        [
            SystemMessage(content="Refrase the user query generating a subquery in norwegian. If the user query har several queries, generate several subqueries. Do not answer the subqueries"),
            HumanMessage(content=f"Here is query from a user: {state['query']}"),
        ]
    )
    return {"subqueries": report_queries.subqueries}

def llm_call(state: WorkerState):
    """Worker answers a subquery using the relevant index"""
    writer = get_stream_writer()  
    writer({"action": f"Worker answers the subquery \"{state['subquery'].subquery}\" using the relevant index"}) 
    
    query_engine = state["query_engine"]
    
    response_obj = query_engine.query(state['subquery'].subquery)
    
    cutoff = state["similarity_cutoff"]
    refs: List[Reference] = []
    found_relevant_reference = False
    
    for node in response_obj.source_nodes:
        if node.score is not None and node.score >= cutoff:
            found_relevant_reference = True
            meta = node.metadata
            refs.append({
                "name": meta.get('title', 'Ingen tittel').lstrip(),
                "url": meta.get('url', 'Ingen URL'),
                "relevancy_index": node.score
            })
    if (found_relevant_reference):
        state['subquery'].response_validity = "valid"
        state['subquery'].references = refs
        state['subquery'].answer = response_obj.response
    else: 
        state['subquery'].response_validity = "not valid"
        state['subquery'].answer = "Jeg beklager, men jeg kan bare svare på spørsmål basert på den gitte konteksten"
        
    return {}

    
def synthesizer(state: State_SubqueryOrchestrator):
    """Synthesize full answer from answers from the subqueries"""
    
    writer = get_stream_writer()  
    writer({"action": "Synthesize full answer from answers from the subqueries"}) 
    
    llm = state["llm"]
    
      # Get the list of SubQuery objects
    sq: list[SubQuery] = state["subqueries"]
    
    completed_report_answers=""
    for s in sq:
        combined = f"# {s.subquery}\n\n"
        combined += f"\n{s.answer}\n\n"
        if s.references:
            combined += "## Referanser\n"
            for r in s.references:
                combined += f"- [{r['name']}]({r['url']}) , relevans: {r['relevancy_index']:.2f}\n"
        completed_report_answers += combined  
    
    # print(f'completed_report: {completed_report_answers}')
        
    aggregated_answer = llm.invoke(
        [
            SystemMessage(
                content="from this list of answers, reorganize a final answer. Use markdown formatting. Do not change the section for Referanser."
            ),
            HumanMessage(
                content=f"Here is the list of answers: {completed_report_answers}"
            ),
        ]
    )
    print(f'Final answer: {aggregated_answer}')
    writer({"final_answer": aggregated_answer.content}) 
    return {"final_answer": aggregated_answer.content}



# Conditional edge function to create llm_call workers that each write a section of the report
def assign_workers(state: State_SubqueryOrchestrator):
    """Assign a worker to each section in the plan"""

    # Kick off section writing in parallel via Send() API
    return [
        Send(
            "llm_call",
            {
                "subquery": s,
                "query_engine": state["query_engine"],
                "similarity_cutoff": state["similarity_cutoff"],
            },
        )
        for s in state["subqueries"]
    ]

# Build workflow
builder = StateGraph(State_SubqueryOrchestrator)

# Add the nodes
builder.add_node("orchestrator", orchestrator)
builder.add_node("llm_call", llm_call)
builder.add_node("synthesizer", synthesizer)

# Add edges to connect nodes
builder.add_edge(START, "orchestrator")
builder.add_conditional_edges(
    "orchestrator", assign_workers, ["llm_call"]
)
builder.add_edge("llm_call", "synthesizer")
builder.add_edge("synthesizer", END)

# Compile the workflow
orchestrator_worker = builder.compile()



