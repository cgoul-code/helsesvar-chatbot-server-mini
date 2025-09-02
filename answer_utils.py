from agent_workflow_structured_answer import (optimizer_workflow, State_StructuredAnswer)
from agent_workflow_answer_with_related_queries import (answer_with_related_queries_workflow, State_AnswerWithRelatedQueries)
from config import ServerSettings, VectorIndexStore, CustomError
from query_utils import QuerySettings
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
    
    # 5) Build the queryengine for related queries
    keywords = {
        "aldersforskjeller",
        "aldersgrense",
        "alkohol og rus",
        "avhengighet av porno",
        "bildedeling",
        "digital blotting",
        "digital sikkerhet",
        "diagnose",
        "digitale overgrep",
        "erogene soner",
        "fantasi versus handling",
        "fetisj",
        "første gang sex",
        "gaming",
        "grenser",
        "grensetråkk",
        "hentai",
        "hva er normalt",
        "hva er pedofili",
        "hvordan gi samtykke",
        "håndtering av grenser",
        "intime grenser",
        "kommunikasjon i forhold",
        "konflikthåndtering",
        "kroppsspråk",
        "kulturelle forventninger",
        "kulturelle perspektiver",
        "lov om deling av nakenbilder",
        "lovlige og ulovlige seksuelle handlinger",
        "maktskjevhet i forhold",
        "nakenbilder",
        "nettvett",
        "onanering",
        "overdrevet onanering",
        "personvern",
        "pornografi",
        "porno og påvirkning",
        "press i relasjoner",
        "pubertet",
        "quiz om seksualitet",
        "religiøse perspektiver",
        "respekt for grenser",
        "rettigheter",
        "rettslige konsekvenser",
        "samtykke",
        "seksualitet",
        "seksualitet i sosiale medier",
        "seksualitet og religion",
        "seksuelle fantasier",
        "seksuelle handlinger",
        "seksuelle rettigheter",
        "seksuell helse",
        "seksuell nysgjerrighet",
        "seksuell trakassering",
        "seksuelt press",
        "selvregulering av seksuell adferd",
        "skam",
        "skyldfølelse",
        "slettmeg.no",
        "sosiale normer og sex",
        "stereotyper",
        "straff for seksuelle overgrep",
        "straff for voldtekt",
        "sunn onanipraksis",
        "tabubelagte tanker",
        "tabuer rundt seksualitet",
        "ulovlig deling",
        "ungdom og seksualitet"
    }
     # 6) define a related question template
    text_rq_template =  ChatPromptTemplate(  [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "You will be provided with a user query.\n"
                "Follow these steps to answer:\n"
                f"Step 1: Classify the query into a maximum of 3 different categories using the following list of keywords: {keywords}.\n"
                "Step 2: For each category from Step 1, use the given context to display 2 relevant questions in Norwegian.\n"
                "Provide the answer using the following JSON format:\n"
                #"[{\"Category name\": \"category\",\"Related questions\":\"question\",}...]\n"
                "[{\"Category name\": \"category\",\"Related questions\":[\"question\",...]},...]\n"
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
        
    response_related_queries_synthesizer = get_response_synthesizer(
        response_mode= "tree_summarize",
        summary_template= text_rq_template, 
        streaming = False,
        structured_answer_filtering=True, 
        verbose=True)

    query_engine_related_queries = index.as_query_engine(
        similarity_cutoff=query_settings.similarity_cutoff,
        similarity_top_k=query_settings.similarity_top_k,
        response_synthesizer=response_related_queries_synthesizer,
    )
    
    
    

    # 5) Initialize and run your optimizer workflow
    init_state: State_AnswerWithRelatedQueries = {
        "llm": server_settings.llm,
        "query_engine": query_engine,
        "query_engine_related_queries": query_engine_related_queries,
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

    final_state = answer_with_related_queries_workflow.invoke(init_state)
    
    from IPython.display import Markdown
    Markdown(final_state["structured_answer"])

    # 6) Return the raw string
    return final_state["structured_answer"]
