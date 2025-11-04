from .agent_workflow_build_hei_index import (build_hei_index_workflow, State_buildIndex)
import logging, os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from dotenv import load_dotenv, find_dotenv
from llama_index.core import Settings

load_dotenv(find_dotenv())

LLMGPT4 = AzureChatOpenAI(
    azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    model_kwargs={"response_format": {"type": "json_object"}},
    #temperature=0.0,
    timeout=120,
)

Settings.embed_model = AzureOpenAIEmbedding(
    model=os.getenv('AZURE_OPENAI_EMBEDDINGS_MODEL'),
    deployment_name=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
    api_key=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_VERSION"),
)

#download_and_persist_storage("hvaerinnafor", "./blobstorage/chatbot/hvaerinnafor" )

logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True)

# 5) Initialize and run workflow
init_state: State_buildIndex = {
    "llm": LLMGPT4,
    "similarity_top_k" : 10,
    "similarity_cutoff" : 0.75,
    "content_type": "html_content",
    "name": "hvaerinnafor",
    "name_questions_answered": "hvaerinnafor_question_bank",
    "storage": "./blobstorage/chatbot/",
    "blobstorage": False,
    "documents": [],
    "documents_text": "",
    "blobstorage": False,
    "keyword_sets": [],   # avoid KeyError
}

final_state = build_hei_index_workflow.invoke(init_state)