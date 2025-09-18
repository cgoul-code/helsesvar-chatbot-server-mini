from .agent_workflow_build_hei_index import (build_hei_index_workflow, State_buildIndex)
import logging, os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

LLMGPT4 = AzureChatOpenAI(
    azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    model_kwargs={"response_format": {"type": "json_object"}},
    #temperature=0.0,
    timeout=120,
)

#download_and_persist_storage("hvaerinnafor", "./blobstorage/chatbot/hvaerinnafor" )

logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True)

# 5) Initialize and run workflow
init_state: State_buildIndex = {
    "llm": LLMGPT4,
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