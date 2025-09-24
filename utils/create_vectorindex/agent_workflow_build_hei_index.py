import os
import re
import logging
import json
import numpy as np
from typing import List, Literal, Optional
from typing_extensions import TypedDict
from sklearn.metrics.pairwise import cosine_similarity
import registry
from registry import severity_for_text_prompt, severity_for_query_prompt, qa_subject_no_prompt, vectorindex_summary_prompt

import pandas as pd
from pandas import json_normalize
from pathlib import Path

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
class KeywordSet(TypedDict):
    Mainkeywords: str
    Keywords: List[str]

class State_buildIndex(TypedDict):
    llm: any  # LLM client (from server_settings.get_llm())
    content_type: Literal["markdown_content", "html_content"]
    name: str
    storage: str
    documents: List[Document]
    answered_questions: List[Document]
    documents_text : str
    blobstorage : bool
    keyword_sets: List[KeywordSet]
    
# === Helpers ===

def _embed_texts(texts: List[str], model: Optional[AzureOpenAIEmbedding] = None, batch_size: int = 64) -> np.ndarray:
    model = model 
    embs = []
    for i in range(0, len(texts), batch_size):
        embs.extend(model.get_text_embedding_batch(texts[i:i+batch_size]))
    return np.asarray(embs, dtype=np.float32)

def _deduplicate_semantic_docs_greedy(
    docs: List[Document],
    embeddings: Optional[np.ndarray] = None,
    threshold: float = 0.85,
    embed_model: Optional[AzureOpenAIEmbedding] = None,
) -> List[Document]:
    """
    Greedy semantic deduplication over a list of Document objects (uses doc.text).
    Keeps the first representative if max similarity < threshold.
    Returns a filtered list of Documents.
    """
    if not docs:
        return []

    texts = [(d.text if isinstance(d.text, str) else "" if d.text is None else str(d.text)) for d in docs]

    if embeddings is None:
        embeddings = _embed_texts(texts, model=embed_model)

    keep_idx: List[int] = []
    for i in range(len(texts)):
        if not keep_idx:
            keep_idx.append(i)
            continue
        sims = cosine_similarity(embeddings[i:i+1], embeddings[keep_idx])[0]
        if float(np.max(sims)) < threshold:
            keep_idx.append(i)

    return [docs[i] for i in keep_idx]

def _persist_storage_for_item(name: str, storage:str, blob_storage: str, documents: List[Document]):
    
    logging.info(f'2 - Loaded {len(documents)} documents')

    try: 
        # Split documents into Node objectts
        #nodes = SentenceSplitter.from_defaults(chunk_size=256, chunk_overlap=75).get_nodes_from_documents(documents)
        # from llama_index.core.node_parser import SentenceSplitter  # or TokenTextSplitter
        # Settings.text_splitter = SentenceSplitter(
        #     chunk_size=1200,
        #     chunk_overlap=150,
        #     )
    # Settings.embed_model = AzureOpenAIEmbedding(
    #     model=os.getenv('AZURE_OPENAI_EMBEDDINGS_MODEL'),
    #     deployment_name=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
    #     api_key=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY"),
    #     azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT"),
    #     api_version=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_VERSION"),
    # )

        # nodes = Settings.text_splitter.get_nodes_from_documents(documents)

        # logging.info ('3 - SentenceSplitter ok')

        # for node in nodes:
        #     logging.info(f'\n\n---Node---:\n{node.metadata}')


        # logging.info(f'4 - Loaded {len(nodes)} nodes')

        # # Create and persist the index
        # storage_context = StorageContext.from_defaults()
        # logging.info('5 - StorageContext.from_defaults ok')

        # index = VectorStoreIndex(nodes, storage_context=storage_context)
        
        storage_context = StorageContext.from_defaults()
        index = VectorStoreIndex.from_documents(
            documents,                      
            storage_context=storage_context,
            show_progress=True,
        )
                
        logging.info('6 - VectorStoreIndex created in memory')
        
        # store locally

        storage_context.persist(persist_dir=storage+name)

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

  
def _iter_meta_rows(meta: dict, base_path: str = ""):
    """
    Yield rows for a 'long' metadata table.
    - For dicts: recurse with dotted keys.
    - For lists: one row per element with meta_index and meta_path like 'answered_questions[0]'.
    - For scalars: single row.
    """
    if meta is None:
        return

    if isinstance(meta, dict):
        for k, v in meta.items():
            path = f"{base_path}.{k}" if base_path else k
            if isinstance(v, (dict, list)):
                yield from _iter_meta_rows(v, base_path=path)
            else:
                # scalar
                yield {
                    "meta_key": k,
                    "meta_index": None,
                    "meta_path": path,
                    "value": v,
                    "value_type": type(v).__name__,
                    "value_len": len(v) if isinstance(v, (str, list, dict)) else None,
                }
    elif isinstance(meta, list):
        for i, v in enumerate(meta):
            path = f"{base_path}[{i}]" if base_path else f"[{i}]"
            if isinstance(v, (dict, list)):
                # expand nested structure
                for row in _iter_meta_rows(v, base_path=path):
                    # Propagate index for top-level lists
                    if row.get("meta_index") is None:
                        row["meta_index"] = i
                    yield row
            else:
                yield {
                    "meta_key": base_path.split(".")[-1] if base_path else "",
                    "meta_index": i,
                    "meta_path": path,
                    "value": v,
                    "value_type": type(v).__name__,
                    "value_len": len(v) if isinstance(v, (str, list, dict)) else None,
                }
    else:
        # scalar at root
        yield {
            "meta_key": base_path or "",
            "meta_index": None,
            "meta_path": base_path or "",
            "value": meta,
            "value_type": type(meta).__name__,
            "value_len": len(meta) if isinstance(meta, (str, list, dict)) else None,
        }

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

    # Extract the title from the doc_item's metadata
    title = doc_item.get('metadata', {}).get('title', 'Untitled')
    keywords =doc_item.get('metadata', {}).get('keywords', 'Untitled')
    description =doc_item.get('metadata', {}).get('description', 'Untitled')

    #print(f'Title: {title}, Keywords: {keywords}, Description: {description}' )
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
        "severity":"",
        "answered_questions":""
        }
    )
    #print(f'\nMetadata: {metadata}')

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

    #print(f'documents_text: {documents_text}')

    print('------------ apify-client run & dataset fetch OK -----------')
    
    return {"documents": documents, "documents_text": documents_text}

def create_metadata_for_documents(state: State_buildIndex) -> dict:

    documents = state["documents"]
    keyword_sets = state.get("keyword_sets", [])
    #print('keyword_sets available to classifier:', keyword_sets)
    updated_docs = []
        
    for doc in documents:
        # sev_prompt = severity_for_text_prompt(doc.text)
        # sev_resp = state["llm"].invoke(sev_prompt)
        # try:
        #     sev_json = json.loads(sev_resp.content)
        #     doc.metadata["severity"] = sev_json.get("category", "")
        # except Exception as e:
        #     print(f"Severity JSON parse error: {e} | raw={sev_resp.content!r}")
        #     doc.metadata["severity"] = ""
        
            
        qa_prompt = qa_subject_no_prompt(text=doc.text)
        qa_resp = state["llm"].invoke(qa_prompt)
        qa_raw = qa_resp.content
        #print('qa raw:', qa_resp.content)
        
        # qa_raw is the string you printed
        questions = []
        try:
            obj = json.loads(qa_raw)           # -> dict
            questions = obj.get("Questions", [])  # -> list[str]
            print('Questions:', questions)
            # validate
            if not isinstance(questions, list) or not all(isinstance(x, str) for x in questions):
                raise ValueError("'Questions' must be a list of strings")
            doc.metadata["answered_questions"] = questions
        except Exception as e:
            print(f"Failed to parse questions: {e} | raw={qa_raw!r}")
            doc.metadata["answered_questions"] = []

        updated_docs.append(doc)
    
    return {"documents": updated_docs}

def create_index_summary_doc(state: State_buildIndex) -> dict:
    
    context = state["documents_text"]
    
    index_summary_prompt = vectorindex_summary_prompt(text=context)
    index_summary_resp = state["llm"].invoke(index_summary_prompt)
    summary_raw = index_summary_resp.content
    
    doc = Document(text = summary_raw)
    
    updated_docs= state["documents"]
    updated_docs.append(doc)
    return {"documents": updated_docs}

def create_answered_questions(state: State_buildIndex) -> dict:
    answered_questions: List[Document] = []

    documents = state["documents"]
    for doc in documents:
        questions = doc.metadata.get("answered_questions", []) or []
        for q in questions:
            new_doc = Document(text=q)
            #new_doc.metadata["severity"] = doc.metadata.get("severity", "")
            new_doc.metadata["from_doc_id"] = getattr(doc, "doc_id", None)
            new_doc.metadata["url"] = doc.metadata.get("url", "")
            print(f'newdoc: {new_doc.text}, {new_doc.metadata}\n')
            answered_questions.append(new_doc)
    print(f'Found {len(answered_questions)} answered questions')
    
    # remove the similar questions 
    unique_answered_questions = _deduplicate_semantic_docs_greedy(answered_questions, embed_model= Settings.embed_model, threshold=0.80)
    
    print(f'Found {len(unique_answered_questions)} unique_answered questions')
    
    # calculated severity for each questions
    for doc in unique_answered_questions:
        sev_prompt = severity_for_query_prompt(query = doc.text)
        sev_resp = state["llm"].invoke(sev_prompt)
        
        try:
            sev_json = json.loads(sev_resp.content)
            doc.metadata["severity"] = sev_json.get("category", "")
        except Exception as e:
            print(f"Severity JSON parse error: {e} | raw={sev_resp.content!r}")
            doc.metadata["severity"] = ""
        
    return {"answered_questions": unique_answered_questions}

def create_log_excel(state: dict) -> dict:
    """
    Writes:
      - 'metadata_long': one row per metadata entry (and per element for lists)
      - 'text': one row per document with full text only
    File: combined_documents_for_{state['name']}.xlsx
    """
    item = state.get("name", "item")
    documents = state.get("documents", [])
    safe_item = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(item))
    out_path = Path(f"combined_documents_for_{safe_item}.xlsx")

    long_rows = []
    text_rows = []

    for idx, doc in enumerate(documents):
        meta = getattr(doc, "metadata", None) or {}
        text = getattr(doc, "text", None)
        text_str = text if isinstance(text, str) else ("" if text is None else str(text))

        # Collect text rows (compact, one per doc)
        text_rows.append({
            "row_index": idx,
            "item": item,
            "text": text_str,
        })

        # Expand metadata into long form
        for row in _iter_meta_rows(meta):
            long_rows.append({
                "row_index": idx,
                "item": item,
                **row
            })

        # Optional console trace
        #print('--------------------------')
        #print(f'doc.metadata: {meta}')
        #preview = (text_str[:200] + "...") if isinstance(text_str, str) and len(text_str) > 200 else text_str
        #print(f'doc.text: {preview}')

    df_long = pd.DataFrame(long_rows, columns=[
        "row_index", "meta_path", "value"
    ])
    df_text = pd.DataFrame(text_rows, columns=["row_index", "item", "text"])

    # Write to Excel
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        df_long.to_excel(writer, sheet_name="metadata_long", index=False)
        df_text.to_excel(writer, sheet_name="text", index=False)

        # Basic formatting
        workbook  = writer.book
        wrap_fmt  = workbook.add_format({"text_wrap": True, "valign": "top"})

        # metadata_long formatting
        ws_long = writer.sheets["metadata_long"]
        for col_idx, col_name in enumerate(df_long.columns):
            width = 60 if col_name in ("value", "meta_path") else 20
            ws_long.set_column(col_idx, col_idx, width, wrap_fmt if col_name in ("value", "meta_path") else None)

        # text formatting
        ws_text = writer.sheets["text"]
        for col_idx, col_name in enumerate(df_text.columns):
            width = 200 if col_name == "text" else 18
            ws_text.set_column(col_idx, col_idx, width, wrap_fmt if col_name == "text" else None)

    print(f"Excel written to: {out_path.resolve()}")
    return {}

def persist_storage(state: State_buildIndex) -> dict:
    documents = state["documents"]
    qa_docs   = state["answered_questions"]
    name      = state["name"]
    storage   = state["storage"]
    blob_on   = state["blobstorage"]
    
    _persist_storage_for_item(name=name, storage=storage, blob_storage=blob_on, documents=documents)
    qa_name = f"{name}_qa_bank"
    _persist_storage_for_item(name=qa_name, storage=storage, blob_storage=blob_on, documents=qa_docs)
   
    return {}



# === Build static, stateless workflow ===
builder = StateGraph(State_buildIndex)

# 1️⃣ Core answer + validation
builder.add_node("apify_call_load_documents", apify_call_load_documents)
builder.add_node("persist_storage", persist_storage)
builder.add_node("create_metadata_for_documents", create_metadata_for_documents)
builder.add_node("create_answered_questions", create_answered_questions)
builder.add_node("create_log_excel", create_log_excel)

builder.add_edge(START, "apify_call_load_documents")
builder.add_edge("apify_call_load_documents", "create_metadata_for_documents")
builder.add_edge("create_metadata_for_documents", "create_answered_questions")
builder.add_edge("create_answered_questions", "create_log_excel")
builder.add_edge("create_log_excel", "persist_storage")

# 5️⃣ Finally, aggregator → END

builder.add_edge("persist_storage", END)

build_hei_index_workflow = builder.compile()

logging.info("answer_witth_related_queries_workflow created...")