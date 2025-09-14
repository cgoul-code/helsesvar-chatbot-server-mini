import os
import re
import logging
import json
from typing import List, Literal
from typing_extensions import TypedDict

from llama_index.core.base.response.schema import Response
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.schema import Document
from llama_index.core import (VectorStoreIndex, StorageContext, Settings)
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

from apify_client import ApifyClient
from typing import Any, Dict, List, Optional, Callable
from dotenv import load_dotenv, find_dotenv
from azure.storage.blob import BlobServiceClient

from langgraph.graph import StateGraph, START, END

MAX_METADATA_LENGTH = 1000

load_dotenv(find_dotenv())


# === Data types ===


class State_buildIndex(TypedDict):
    llm: any  # LLM client (from server_settings.get_llm())
    content_type: Literal["markdown_content", "html_content"]
    name: str
    storage: str
    documents: List[Document]
    documents_text : str
    blobstorage : bool


# === Helpers ===


def _load_run_input(item: str) -> Optional[Dict[str, Any]]:
    """
    Loads run input for the actor (e.g., startUrls, pseudo-URLs, link selectors, etc.)
    """
    run_input_params = _load_configuration(
        item,
        f'./scraping/{item}/config.json',
        f'./scraping/{item}/startUrls.json'
    )
    return run_input_params.get(item, {})

def _call_website_content_crawler(client: ApifyClient, run_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs apify/website-content-crawler and returns the run dict.
    """
    # This starts the actor and waits for it to finish
    run = client.actor("apify/website-content-crawler").call(run_input=run_input)
    return run

def _dataset_items(client: ApifyClient, dataset_id: str, limit: int | None = None):
    ds = client.dataset(dataset_id)
    items_all = []
    for item in ds.iterate_items():  # streams through all pages
        items_all.append(item)
        if limit is not None and len(items_all) >= limit:
            break
    return items_all

def _load_configuration(item, config_file, start_urls_file):
    # Load the main configuration
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
        #print(f'config:{config}')
    
    # Load the startUrls
    with open(start_urls_file, 'r', encoding='utf-8') as f:
        start_urls = json.load(f)
        #print(f'start_urls: {start_urls}')
    
    # Assign the startUrls to the config
    config[item]['startUrls'] = start_urls
    
    return config

def _truncate_metadata(metadata: dict) -> dict:
    # Set max lengths per field based on priority
    max_lengths = {
        "url": 300,
        "title": 200,
        "keywords": 300,
        "description": 300,
    }

    truncated = {}
    total_length = 2  # account for braces and commas

    for key in ["url", "title", "keywords", "description"]:
        value = str(metadata.get(key, ""))
        max_len = max_lengths.get(key, 200)
        value = value[:max_len]
        truncated[key] = value
        total_length += len(key) + len(value) + 4  # for formatting

    # Re-check total length and cut more if needed
    while True:
        json_len = len(str(truncated))
        if json_len <= MAX_METADATA_LENGTH:
            break
        # Trim the least important field
        for field in ["description", "keywords", "title", "url"]:
            if len(truncated[field]) > 50:
                truncated[field] = truncated[field][:-10]
                break
        else:
            break

    return truncated

def _transform_dataset_item(doc_item, content_type):
    # Extract HTML content from the scraped doc_item
    print(f'---------{content_type}')

    # Extract the title from the doc_item's metadata
    title = doc_item.get('metadata', {}).get('title', 'Untitled')
    keywords =doc_item.get('metadata', {}).get('keywords', 'Untitled')
    description =doc_item.get('metadata', {}).get('description', 'Untitled')

    print(f'Title: {title}, Keywords: {keywords}, Description: {description}' )
    markdown_content = doc_item.get("markdown", "")
    if(content_type=="markdown_content"):
        text_content = doc_item.get("markdown", "")
    else:
        text_content = doc_item.get("text", "")
    
    # remove all text after split_phrase
    split_phrase = "Fikk du svar på det du lurte på?" 
    result = text_content.partition(split_phrase)[0]

    # fix for getting correct title from Hva er innafor:
    if(title=="Hva er innafor - Helsenorge"):
        # Regular expression to capture the text after "Spørsmål:"
        match = re.search(r"Spørsmål:\s*(.*)", markdown_content, re.DOTALL)

        # Extracting and printing the result
        if match:
            title = match.group(1).splitlines()[0]
            print(f'title fra spørsmål:{title}')
        else:
            print("No 'Spørsmål' section found.")


    # Extract the URL from the doc_item
    url = doc_item.get("url", "Unknown URL")

    # Create metadata including the title and URL
    metadata = _truncate_metadata(
        {
        "url": url, 
        "title": title,
        "keywords": keywords,
        "description": description,
        }
    )
    print(f'\nMetadata: {metadata}')

    # Return the transformed item as a Document with metadata
    return Document(text=result, metadata=metadata)

# === Node functions ===

def apify_call_load_documents(state: State_buildIndex) -> dict:
    """
    Scrape using Apify's official client (apify-client), run the 'apify/website-content-crawler'
    actor with your config/startUrls, fetch dataset items, then map them through your
    transform_dataset_item with a content_type hint.
    """
    item = state["name"]
    content_type = state["content_type"]

    print(f'documents_from_startUrls, item: {item}')

    run_input = None
    if item in ['hvaerinnafor']:
        run_input = _load_run_input(item)

    if not run_input:
        print("Run input not found!")
        return []


    # Init client — prefers APIFY_TOKEN; falls back to APIFY_KEY if that's what you already use
    
    token = os.getenv('APIFY_TOKEN')
    if not token:
        raise RuntimeError("Missing APIFY_TOKEN in environment.")

    client = ApifyClient(token)

    # Run the actor
    run = _call_website_content_crawler(client, run_input)
    dataset_id = run.get("defaultDatasetId")
    if not dataset_id:
        print("No defaultDatasetId found on run result.")
        return []

    # Pull items and transform
    raw_items = _dataset_items(client, dataset_id)
    transform_with_content_type: Callable[[Dict[str, Any]], Any] = (
        lambda it: _transform_dataset_item(it, content_type=content_type)
    )
    documents = [transform_with_content_type(it) for it in raw_items]
    
    documents_text = ""
    for doc in documents:
        documents_text += doc.metadata["title"]
        documents_text += doc.text

    print(f'documents_text: {documents_text}')

    print('------------ apify-client run & dataset fetch OK -----------')
    
    return {"documents": documents, "documents_text": documents_text}

def create_key_words(state: State_buildIndex) -> dict:

    msg = state["llm"].invoke(
        f'Based on the text I provide you, generate a list of relevant keywords and related questions: {state["documents_text"]}'
        ' Output requirements:\n'
        '- Return JSON ONLY. No explanations. No markdown. No code fences.\n'
        '- Use double quotes for all keys and strings.\n'
        '- Do not include trailing commas.\n'
        '- Avoid duplicate keywords.\n'
        '- Shape:\n'
        '[{{"Keywords": "category", "Related questions": ["question1", "question2"]}}, ...]\n'
    )
    print(f'keywords: {msg}')

    return{}


def persist_storage(state: State_buildIndex) -> dict:

    documents = state["documents"]
    name = state["name"]

    logging.info(f'2 - Loaded {len(documents)} documents')

    try: 
        # Split documents into Node objectts
        #nodes = SentenceSplitter.from_defaults(chunk_size=256, chunk_overlap=75).get_nodes_from_documents(documents)
        # from llama_index.core.node_parser import SentenceSplitter  # or TokenTextSplitter
        # Settings.text_splitter = SentenceSplitter(
        #     chunk_size=1200,
        #     chunk_overlap=150,
        #     )
        print (f'inside try-catch')
        Settings.embed_model = AzureOpenAIEmbedding(
            model=os.getenv('AZURE_OPENAI_EMBEDDINGS_MODEL'),
            deployment_name=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
            api_key=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_VERSION"),
        )

        nodes = Settings.text_splitter.get_nodes_from_documents(documents)

        logging.info ('3 - SentenceSplitter ok')

        for node in nodes:
            logging.info(f'\n\n---Node---:\n{node.metadata}')


        logging.info(f'4 - Loaded {len(nodes)} nodes')

        # Create and persist the index
        storage_context = StorageContext.from_defaults()
        logging.info('5 - StorageContext.from_defaults ok')

        index = VectorStoreIndex(nodes, storage_context=storage_context)
        logging.info('6 - VectorStoreIndex created in memory')
        
        # store locally
        storage = state["storage"]
        storage_context.persist(persist_dir=storage)
        
        blob_storage = state["blobstorage"]

        if (blob_storage):

            LOCAL_STORAGE_PATH = "vector-index"  # Default for local testing
            # Ensure the folder exists
            os.makedirs(LOCAL_STORAGE_PATH, exist_ok=True)
            # running on Azure
            storage_context.persist(persist_dir=LOCAL_STORAGE_PATH)

            # copy files from local storage to blobcontainer
            # Get environment variables for connection and container
            connection_string = os.getenv('CONNECTION_STRING')
            container_name = os.getenv('CONTAINER_NAME')

            # Directory containing files on azure 
            local_directory = "/home/vector-index"

            # Initialize the BlobServiceClient
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)

            # Iterate through files in the directory
            for filename in os.listdir(local_directory):
                local_file_path = os.path.join(local_directory, filename)

                # Ensure it's a file (not a directory)
                if os.path.isfile(local_file_path):
                    # Define the path inside the container
                    blob_path = f"{name}/{filename}"  
                    logging.info(f'Blob_path is {blob_path}')
                    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_path)

                    # Upload the file
                    with open(local_file_path, "rb") as data:
                        blob_client.upload_blob(data, overwrite=True)

                    logging.info(f"File {local_file_path} uploaded successfully to {container_name}/{blob_path}")

                    # Optionally, delete the local file after upload
                    os.remove(local_file_path)
                    logging.info(f"Deleted local file: {local_file_path}")



        logging.info('7 - storage_context.persist ok')
        return {}

    except Exception:
        logging.error("Failed to persist storage:\n%s", exc_info=True)
        # Fail-soft: keep pipeline alive
        return {}

  


# === Build static, stateless workflow ===
builder = StateGraph(State_buildIndex)

# 1️⃣ Core answer + validation
builder.add_node("apify_call_load_documents", apify_call_load_documents)
builder.add_node("persist_storage", persist_storage)
builder.add_node("create_key_words", create_key_words)




builder.add_edge(START, "apify_call_load_documents")
builder.add_edge("apify_call_load_documents", "create_key_words")

# 5️⃣ Finally, aggregator → END
builder.add_edge("create_key_words", "persist_storage")
builder.add_edge("persist_storage", END)

build_hei_index_workflow = builder.compile()

logging.info("answer_witth_related_queries_workflow created...")