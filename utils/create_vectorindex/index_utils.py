import logging
import os, json, re
from dotenv import load_dotenv, find_dotenv
import time
from llama_index.core import (VectorStoreIndex, StorageContext, Settings,  load_index_from_storage)
from llama_index.embeddings.openai import OpenAIEmbedding
import llama_index.core.readers as readers
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from apify_client import ApifyClient
from llama_index.core.schema import Document
from typing import Any, Dict, List, Optional, Callable

MAX_METADATA_LENGTH = 1000

load_dotenv(find_dotenv())

# Check if running in Azure Web App and set the correct local storage path
if os.getenv("WEBSITE_SITE_NAME"):  # This environment variable exists in Azure App Services
    LOCAL_STORAGE_PATH = "/home/vector-index"  # Linux
else:
    LOCAL_STORAGE_PATH = "vector-index"  # Default for local testing

# Ensure the folder exists
os.makedirs(LOCAL_STORAGE_PATH, exist_ok=True)


def _pick_content_type(item: str) -> str:
    """
    Decide which content type we want the transform to receive.
    """
    if item in {"hvilkekiregelverk", "rapportbrukavki", "ungnospmtobakk"}:
        return "markdown_content"
    return "html_content"

def _load_run_input(item: str) -> Optional[Dict[str, Any]]:
    """
    Loads run input for the actor (e.g., startUrls, pseudo-URLs, link selectors, etc.)
    """
    run_input_params = load_configuration(
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




def truncate_metadata(metadata: dict) -> dict:
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

def transform_dataset_item(item, content_type):
    # Extract HTML content from the scraped item
    print(f'---------{content_type}')

        # Extract the title from the item's metadata
    title = item.get('metadata', {}).get('title', 'Untitled')
    keywords =item.get('metadata', {}).get('keywords', 'Untitled')
    description =item.get('metadata', {}).get('description', 'Untitled')

    print(f'Title: {title}, Keywords: {keywords}, Description: {description}' )
    markdown_content = item.get("markdown", "")
    if(content_type=="markdown_content"):
        text_content = item.get("markdown", "")
    else:
        text_content = item.get("text", "")
    
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


    # Extract the URL from the item
    url = item.get("url", "Unknown URL")

    # Create metadata including the title and URL
    metadata = truncate_metadata(
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

def load_configuration(item, config_file, start_urls_file):
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

def documents_from_startUrls(item: str) -> List[Any]:
    """
    Scrape using Apify's official client (apify-client), run the 'apify/website-content-crawler'
    actor with your config/startUrls, fetch dataset items, then map them through your
    transform_dataset_item with a content_type hint.
    """
    print(f'documents_from_startUrls, item: {item}')

    run_input = None
    if item in ['hvaerinnafor']:
        run_input = _load_run_input(item)

    if not run_input:
        print("Run input not found!")
        return []

    content_type = _pick_content_type(item)

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
        lambda it: transform_dataset_item(it, content_type=content_type)
    )
    documents = [transform_with_content_type(it) for it in raw_items]

    print('------------ apify-client run & dataset fetch OK -----------')
    return documents

def download_documents_for_item(item):

    
    print('11')
   
    documents: List[Any] = []
    
    if item == "hvaerinnafor":
        documents = documents_from_startUrls(item)

    print(f'Loaded {len(documents)} documents')

    # Create a log for each item
    with open(f'combined_documents_for_{item}.txt', 'w', encoding='utf-8') as file:
        print('--------------------------')
        for doc in documents:
            # Robust printing for console
            print(f'doc.metadata: {doc.metadata}')
            print(f'doc.text: {(doc.text[:200] + "...") if isinstance(doc.text, str) and len(doc.text) > 200 else doc.text}')

            # Serialize metadata as JSON
            meta_json = json.dumps(doc.metadata or {}, ensure_ascii=False)
            file.write(meta_json + '\n')

            # Ensure text is a string
            text_str = doc.text if isinstance(doc.text, str) else ("" if doc.text is None else str(doc.text))
            file.write(text_str + '\n\n')

    return documents

def download_and_persist_storage(name, storage):
    logging.info('1 - download_and_persist_storage')
    documents = download_documents_for_item(name)
    print (f'2 - Loaded {len(documents)} documents')

    # Split documents into Node objectts
    #nodes = SentenceSplitter.from_defaults(chunk_size=256, chunk_overlap=75).get_nodes_from_documents(documents)
    Settings.embed_model = OpenAIEmbedding(model=os.getenv('AZURE_OPENAI_EMBEDDINGS_MODEL'))
    nodes = Settings.text_splitter.get_nodes_from_documents(documents)
    logging.info ('3 - SentenceSplitter ok')

    for node in nodes:
        logging.info(f'\n\n---Node---:\n{node.metadata}')

    logging.info('3 - Splitting nodes OK')
    logging.info(f'4 - Loaded {len(nodes)} nodes')

    # Create and persist the index
    storage_context = StorageContext.from_defaults()
    logging.info('5 - StorageContext.from_defaults ok')

    index = VectorStoreIndex(nodes, storage_context=storage_context)
    logging.info('6 - VectorStoreIndex created in memory')

    # if ('WEBSITE_SITE_NAME' in os.environ or 'FUNCTIONS_WORKER_RUNTIME' in os.environ):
    #     # running on Azure
    #     storage_context.persist(persist_dir=LOCAL_STORAGE_PATH)

    #     # copy files from local storage to blobcontainer
    #     # Get environment variables for connection and container
    #     connection_string = os.getenv('CONNECTION_STRING')
    #     container_name = os.getenv('CONTAINER_NAME')

    #     # Directory containing files on azure 
    #     local_directory = "/home/vector-index"

    #     # Initialize the BlobServiceClient
    #     blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    #     # Iterate through files in the directory
    #     for filename in os.listdir(local_directory):
    #         local_file_path = os.path.join(local_directory, filename)

    #         # Ensure it's a file (not a directory)
    #         if os.path.isfile(local_file_path):
    #             # Define the path inside the container
    #             blob_path = f"{name}/{filename}"  
    #             logging.info(f'Blob_path is {blob_path}')
    #             blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_path)

    #             # Upload the file
    #             with open(local_file_path, "rb") as data:
    #                 blob_client.upload_blob(data, overwrite=True)

    #             logging.info(f"File {local_file_path} uploaded successfully to {container_name}/{blob_path}")

    #             # Optionally, delete the local file after upload
    #             os.remove(local_file_path)
    #             logging.info(f"Deleted local file: {local_file_path}")


    # else:
        # running local

        # persist locally only (you need to copy the files manually to the blobcontainer)
    storage_context.persist(persist_dir=storage)


    logging.info('7 - storage_context.persist ok')

    return index