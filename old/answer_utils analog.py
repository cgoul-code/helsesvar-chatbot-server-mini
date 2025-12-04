from old.agent_workflow_structured_answer import (optimizer_workflow, State_StructuredAnswer)
from agent_workflow_answer import (answer_with_related_queries_workflow, State_AnswerWithRelatedQueries)
from config import ServerSettings, VectorIndexStore, CustomError
from query_utils import QuerySettings
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.core.query_engine import RetrieverQueryEngine 
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate, get_response_synthesizer, VectorStoreIndex
import logging
import IPython



def get_structured_answer(
    query_settings: QuerySettings,
    server_settings: ServerSettings,
    vector_store: VectorIndexStore
) -> str:
    print('------------->>get_structured_answer')
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
        streaming=True,
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
        "structured_answer": "",
    }

    final_state = optimizer_workflow.invoke(init_state)
    
    from IPython.display import Markdown
    Markdown(final_state["structured_answer"])

    # 6) Return the raw string
    return final_state["structured_answer"]

#----------------------------------------------------------

def get_answer_with_related_queries(
    query_settings: QuerySettings,
    server_settings: ServerSettings,
    vector_store: VectorIndexStore
) -> str:
    print('------------->>get_answer_with_related_queries')
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
    vec_name_qa_bank = f'{query_settings.vectorIndex}_qa_bank'
    entry_qa_bank = vector_store.get(vec_name_qa_bank)
    if entry_qa_bank is None:
        # Log with %s formatting
        logging.error("Index not found: %s", vec_name_qa_bank)
        # Raise so the route handler can catch & return 404 JSON
        raise CustomError(
            f"Index not found, referansefilene for {vec_name_qa_bank} mangler!",
            404
        )

    index_qa_bank: VectorStoreIndex = entry_qa_bank.index
    vector_index_description_qa_bank = entry_qa_bank.description

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
                "Given the context information and not prior knowledge, "
                "Query: {query_str}\n"
                "Answer: "
            ),
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
    
     # 6) define a related question template
    text_rq_template = ChatPromptTemplate([
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                # "You will be provided with a user query.\n"
                # "From the user query, find 5 queries that kan be related to the user query"
                # "Output requirements:\n"
                # "- Return JSON ONLY. No explanations. No markdown. No code fences.\n"
                # "- Use double quotes for all keys and strings.\n"
                # "- Do not include trailing commas.\n"
                # "- Keep information of the context for from the initial user query, an example: if the user query concerns the sharing of nude images, and a relevant query is about penalties, the relevant question should address penalties for sharing nude images.\n"
                # "- Shape:\n"
                # '[{"Category name": "category","Related questions":["question1","question2"]}, ...]\n'
                
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
    init_state: State_AnswerWithRelatedQueries = {
        "llm": server_settings.llm,
        "query_engine": query_engine,
        "query_engine_related_queries": query_engine_related_queries,
        "retriever_related_queries": retriever_related_queries,
        "vector_index_description": vector_index_description,
        "query": query_settings.user_content,
        "similarity_cutoff": query_settings.similarity_cutoff,
        "relevancy_cutoff" : query_settings.relevancy_cutoff,
        # defaults:
        "response": None,
        "answer": "",
        "lix_score": 0.0,
        "lix_category": "",
        "readable_or_not": "not readable",
        "feedback": "",
        "references": [],
        "structured_answer": "",
    }

    final_state = answer_with_related_queries_workflow.invoke(init_state)
    
    from IPython.display import Markdown
    Markdown(final_state["structured_answer"])

    # 6) Return the raw string
    return final_state["structured_answer"]
