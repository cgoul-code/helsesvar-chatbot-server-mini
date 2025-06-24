from agent_workflow_structured_answer import (optimizer_workflow, State_StructuredAnswer)
from agent_workflow_subquery_orchestrator import (orchestrator_worker, State_SubqueryOrchestrator)

from config import ServerSettings, VectorIndexStore, CustomError
from query_utils import QuerySettings
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate, get_response_synthesizer, VectorStoreIndex
import logging
import anyio

async def get_answer_structured_as_stream(
    query_settings: QuerySettings,
    server_settings: ServerSettings,
    vector_store: VectorIndexStore
):
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

    # 2) Build your prompt template
    text_qa_template = ChatPromptTemplate([

        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "You are 'HelseSvar', a friendly, empathetic, and knowledgeable health advisor from helsenorge.no, specifically designed to help young people in Norway (ages 13-19).\n\n"
                "Your primary goal is to provide clear, supportive, and easy-to-understand answers to their health questions.\n\n"
                "**Tone and Style Guidelines:**\n"
                "1.  **Empathy:** Always respond with understanding and support. Acknowledge the user's feelings if they express worry, confusion, or distress (e.g., 'I understand this can be a concern,' or 'It's normal to have questions about this.'). Be reassuring and non-judgmental.\n"
                "2.  **Teen-Friendly Language (Ages 13-19):** \n"
                "    - Explain things clearly and directly. Avoid overly medical jargon or complex terminology. If you must use a technical term, explain it immediately in simple words.\n"
                "    - Use short sentences and paragraphs. Break down complex information.\n"
                "    - Maintain a friendly, approachable, and encouraging tone. Imagine you're talking to a smart but not yet expert high school student.\n"
                "    - Example of simplification: Instead of 'The symptomatology typically manifests as...', say 'Usually, you might notice symptoms like...'.\n\n"
                "**Core Rules for Answering:**\n"
                "- Always answer the request using ONLY the provided context information. Do not use any prior knowledge.\n"
                "- Provide detailed explanations from the context, but avoid unnecessary repetitions.\n"
                "- Always answer in Norwegian (Bokmål).\n"
                "- If the context doesn't cover the question, clearly state that the information isn't available in the provided articles."
            )
        ),

        ChatMessage(
            role=MessageRole.USER,
            content=(
                "Context information is below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Query: {query_str}\n"
                "Answer: "
            )
        ),
    ])

    # 3) Create the response synthesizer
    response_synthesizer = get_response_synthesizer(
        response_mode=query_settings.response_mode,
        text_qa_template=text_qa_template,
        summary_template=text_qa_template,
        structured_answer_filtering=True,
        verbose=True,
    )

    # 4) Configure the query engine
    query_engine = index.as_query_engine(
        similarity_cutoff=query_settings.similarity_cutoff,
        similarity_top_k=query_settings.similarity_top_k,
        response_synthesizer=response_synthesizer,
    )

    # 5) Initialize and run your optimizer workflow
    init_state: State_StructuredAnswer = {
        "llm": server_settings.llm,
        "query_engine": query_engine,
        "vector_index_description": vector_index_description,
        "query": query_settings.user_content,
        "similarity_cutoff": query_settings.similarity_cutoff,
        # defaults:
        "response": None,
        "answer": "",
        "lix_score": 0.0,
        "lix_category": "",
        "readable_or_not": "not readable",
        "feedback": "",
        "references": [],
        "query_short_version": "", 
        "query_summary": "",
        "final_answer": "",
        "num_iterations": 0,
    }
    

    # ✅ This is  an **async generator**
    async for chunk in optimizer_workflow.astream(init_state, stream_mode="custom"):
        yield chunk


async def get_answer_with_subqueries_as_stream(
    query_settings: QuerySettings,
    server_settings: ServerSettings,
    vector_store: VectorIndexStore
):
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

    # 2) Build your prompt template
    text_qa_template = ChatPromptTemplate([

        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "You are 'HelseSvar', a friendly, empathetic, and knowledgeable health advisor from helsenorge.no, specifically designed to help young people in Norway (ages 13-19).\n\n"
                "Your primary goal is to provide clear, supportive, and easy-to-understand answers to their health questions.\n\n"
                "**Tone and Style Guidelines:**\n"
                "1.  **Empathy:** Always respond with understanding and support. Acknowledge the user's feelings if they express worry, confusion, or distress (e.g., 'I understand this can be a concern,' or 'It's normal to have questions about this.'). Be reassuring and non-judgmental.\n"
                "2.  **Teen-Friendly Language (Ages 13-19):** \n"
                "    - Explain things clearly and directly. Avoid overly medical jargon or complex terminology. If you must use a technical term, explain it immediately in simple words.\n"
                "    - Use short sentences and paragraphs. Break down complex information.\n"
                "    - Maintain a friendly, approachable, and encouraging tone. Imagine you're talking to a smart but not yet expert high school student.\n"
                "    - Example of simplification: Instead of 'The symptomatology typically manifests as...', say 'Usually, you might notice symptoms like...'.\n\n"
                "**Core Rules for Answering:**\n"
                "- Always answer the request using ONLY the provided context information. Do not use any prior knowledge.\n"
                "- Provide detailed explanations from the context, but avoid unnecessary repetitions.\n"
                "- Always answer in Norwegian (Bokmål).\n"
                "- If the context doesn't cover the question, clearly state that the information isn't available in the provided articles."
            )
        ),

        ChatMessage(
            role=MessageRole.USER,
            content=(
                "Context information is below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Query: {query_str}\n"
                "Answer: "
            )
        ),
    ])

    # 3) Create the response synthesizer
    response_synthesizer = get_response_synthesizer(
        response_mode=query_settings.response_mode,
        text_qa_template=text_qa_template,
        summary_template=text_qa_template,
        structured_answer_filtering=True,
        verbose=True,
    )

    # 4) Configure the query engine
    query_engine = index.as_query_engine(
        similarity_cutoff=query_settings.similarity_cutoff,
        similarity_top_k=query_settings.similarity_top_k,
        response_synthesizer=response_synthesizer,
    )
    
    # 5) Initialize and run your optimizer workflow
    init_state: State_SubqueryOrchestrator = {
        "llm": server_settings.llm,
        "query_engine": query_engine,
        "query": query_settings.user_content,
        "similarity_cutoff": query_settings.similarity_cutoff,
        # defaults:
        "subqueries": [],

    }

    # ✅ This is  an **async generator**
    async for chunk in orchestrator_worker.astream(init_state, stream_mode="custom"):
        yield chunk