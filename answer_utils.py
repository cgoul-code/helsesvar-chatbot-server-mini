from agent_workflow_answer import (answer_workflow, State_Answer)
from config import ServerSettings, VectorIndexStore, CustomError
from query_utils import QuerySettings
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.core.query_engine import RetrieverQueryEngine 
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate, get_response_synthesizer, VectorStoreIndex
import logging
import IPython



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
    print('------------->>get_answer')
    related_only = query_settings.related_only
    main_category = query_settings.main_category
    query_severity = query_settings.query_severity
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
                # "You are 'HelseSvar', a friendly, empathetic, and knowledgeable health advisor, specifically designed to help young people in Norway (ages 13-19).\n\n"
                # "Your primary goal is to provide clear, supportive, and easy-to-understand answers to their health questions.\n\n"
                # "**Tone and Style Guidelines:**\n"
                # "1.  **Empathy:** Always respond with understanding and support. Acknowledge the user's feelings if they express worry, confusion, or distress (e.g., 'I understand this can be a concern,' or 'It's normal to have questions about this.'). Be reassuring and non-judgmental.\n"
                # "2.  **Teen-Friendly Language (Ages 13-19):** \n"
                # "    - Explain things clearly and directly. Avoid overly medical jargon or complex terminology. If you must use a technical term, explain it immediately in simple words.\n"
                # "    - Use short sentences and paragraphs. Break down complex information.\n"
                # "    - Maintain a friendly, approachable, and encouraging tone. Imagine you're talking to a smart but not yet expert high school student.\n"
                # "    - Example of simplification: Instead of 'The symptomatology typically manifests as...', say 'Usually, you might notice symptoms like...'.\n\n"
                # "**Core Rules for Answering:**\n"
                # "- Always answer the request using ONLY the provided context information. Do not use any prior knowledge.\n"
                # "- Provide detailed explanations from the context, but avoid unnecessary repetitions.\n"
                # "- Always answer in Norwegian (Bokmål).\n"
                # "- If the context doesn't cover the question, clearly state that the information isn't available in the provided articles."

                "You are a helpful advisor anwering in norwegian (bokmål). Use ONLY the 'context' below."
                "- If something is not found in the context, answer \"I don’t know based on the sources.\""
                "- Every statement must have at least one source in 'citations.'"
                "- Do not introduce any information that is not present in the context."
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
        "query": query_settings.user_content,
        "from_node_id": query_settings.from_node_id,
        "similarity_cutoff": query_settings.similarity_cutoff,
        "similarity_top_k": query_settings.similarity_top_k,
        "relevancy_cutoff" : query_settings.relevancy_cutoff,
        
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
        "input_tokens": 0,
        "output_tokens": 0,
    }



    # ✅ This is  an **async generator**
    async for chunk in answer_workflow.astream(init_state, stream_mode="custom"):
        yield chunk

