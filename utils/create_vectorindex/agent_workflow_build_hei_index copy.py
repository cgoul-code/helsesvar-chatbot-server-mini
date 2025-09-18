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
class KeywordSet(TypedDict):
    Mainkeywords: str
    Keywords: List[str]

class State_buildIndex(TypedDict):
    llm: any  # LLM client (from server_settings.get_llm())
    content_type: Literal["markdown_content", "html_content"]
    name: str
    storage: str
    documents: List[Document]
    questions_answered: List[Document]
    documents_text : str
    blobstorage : bool
    keyword_sets: List[KeywordSet]
    
# === Helpers ===

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
        Settings.embed_model = AzureOpenAIEmbedding(
            model=os.getenv('AZURE_OPENAI_EMBEDDINGS_MODEL'),
            deployment_name=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
            api_key=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_VERSION"),
        )

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
    - For lists: one row per element with meta_index and meta_path like 'answeredQuestions[0]'.
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
        "severity":"",
        "questionsAnswered":""
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

    #print(f'documents_text: {documents_text}')

    print('------------ apify-client run & dataset fetch OK -----------')
    
    return {"documents": documents, "documents_text": documents_text}

def create_metadata_for_documents(state: State_buildIndex) -> dict:
    
        # •	tema: ["Forhold", "Forelskelse", "samtykke", "porno", ...]
        # •	undertema: ["grenser", "tillit", "brudd", "sjalusi", ...] (eksempel på undertemaer for "Forhold")
        # •	målgruppe: ["ungdom", "foreldre", "fagpersoner"]
        # •	aldersgruppe: ["13-15", "16-18", "18+"]
        # •	hensikt: ["informasjon", "hjelpesøkende", "egenvurdering", "krise"]
        # •	tone: ["nøytral", "empatisk", "autoritativ"]

    
    documents = state["documents"]
    keyword_sets = state.get("keyword_sets", [])
    print('keyword_sets available to classifier:', keyword_sets)
    updated_docs = []
        
    for doc in documents:
        sev_prompt = (
            f'You are given a text: {doc.text}\n\n'
            'Categorize the severity of this text into one of three categories: Green, Yellow, or Red.\n\n'

            'Green category:\n'
            '- Preventive and safety-promoting.\n'
            '- Texts that provide general information, knowledge, and guidance to prevent problems and strengthen good sexual health.\n'
            '- The content helps increase understanding, safety, and awareness (e.g., consent, contraception, communication, emotions, body knowledge).\n'
            '- No acute situation or personal crisis is described.\n'
            'Example: "How to talk with your partner about boundaries" or "Facts about condoms".\n\n'

            'Yellow category:\n'
            '- Challenges or vulnerable situations.\n'
            '- Texts that describe concerns, difficulties, or risks that may require reflection or support, but are not acute or immediately dangerous.\n'
            '- May involve difficult feelings, uncertainty in relationships, unwanted experiences, or the need for advice beyond general information.\n'
            '- The reader may need to seek help or guidance, but the situation is not considered an acute crisis.\n'
            'Example: "What should I do if my partner doesn’t respect my boundaries?", '
            '"I regret sending a nude", or topics like "pornography", "sexual pressure", "issues around consent", "(illegal) fetishes".\n\n'

            'Red category:\n'
            '- Serious or acute situations.\n'
            '- Texts that concern serious incidents or crises where the person involved may be in danger or at significant risk of harm.\n'
            '- Includes violence, abuse, coercion, acute psychological crises, or other situations that require immediate follow-up or professional help.\n'
            '- The main purpose of the text is to provide information about where and how to get help quickly.\n'
            'Example: "Sex with animals", "Sex with family members", "Downloading child pornography", "Illegal image sharing".\n\n'

            'Output requirements:\n'
            '- Return JSON ONLY. No explanations. No markdown. No code fences.\n'
            '- Use double quotes for all keys and strings.\n'
            '- Do not include trailing commas.\n'
            '- Output shape:\n'
            '{ "category": "<Green|Yellow|Red>" }'
        )
        sev_resp = state["llm"].invoke(sev_prompt)
        try:
            sev_json = json.loads(sev_resp.content)
            doc.metadata["severity"] = sev_json.get("category", "")
        except Exception as e:
            print(f"Severity JSON parse error: {e} | raw={sev_resp.content!r}")
            doc.metadata["severity"] = ""
        
        
        # kw_prompt = (
        #     f'You are given a text: {doc.text}\n\n'
        #     f'From the given text, identify relevant "Mainkeywords" and "Keywords" ONLY from this set: {keyword_sets}\n\n'
        #     'Output requirements:\n'
        #     '- Return JSON ONLY. No explanations. No markdown. No code fences.\n'
        #     '- Use double quotes for all keys and strings.\n'
        #     '- Do not include trailing commas.\n'
        #     '- Use only mainkeywords and keywords from the provided keyword sets.\n'
        #     '- Shape:\n'
        #     '[{"Mainkeywords": "main", "Keywords": ["k1", "k2"]}, ...]\n'
        # )
        # kw_resp = state["llm"].invoke(kw_prompt)
        # print('kw raw:', kw_resp.content)
        # try:
        #     kw_json = json.loads(kw_resp.content)
        #     if not isinstance(kw_json, list):
        #         raise ValueError("keywords JSON must be a list")
        #     doc.metadata["keyword_sets"] = kw_json
        # except Exception as e:
        #     print(f"Keyword JSON parse error: {e} | raw={kw_resp.content!r}")
        #     doc.metadata["keyword_sets"] = []
            
        qa_prompt = (f'Here is the context: {doc.text}\n'
            'Given the contextual information, \n'
            'generate a list of questions in Norwegian that this context can provide specific answers to, which are unlikely to be found elsewhere.\n'
            '\n'
            'STRICT REQUIREMENT:\n'
            '- Every single question MUST explicitly mention the subject of the context (for example "Forelskelse") instead of referring to "teksten", "artikkelen", "avsnittet" or similar.\n'
            '- Do not use phrases like "ifølge teksten", "hva sier teksten", "nevnes i teksten" etc.\n'
            '- Instead, directly phrase the questions around the subject matter itself.\n\n'
            'Example of WRONG question: "Hva sier teksten om hvordan man merker at man er forelsket?"\n'
            'Example of RIGHT question: "Hva er vanlige tegn på forelskelse som skiller det fra å bare være betatt?"\n\n'
            'Output requirements:\n'
            '- Only generate questions about the subject matter and content of the text\n'
            '- Return JSON ONLY. No explanations. No markdown. No code fences.\n'
            '- Use double quotes for all keys and strings.\n'
            '- Do not include trailing commas.\n'
            'Do NOT generate questions about:\n'
            '- who wrote the article\n'
            '- contributors or authors\n'
            '- which website, publication, or source the text comes from\n'
            '- metadata such as publishing date, copyright, or layout\n\n'
            '- Shape:\n'
            '{"Questions": ["question1", "question2"]}'
            )

        
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
            doc.metadata["answeredQuestions"] = questions
        except Exception as e:
            print(f"Failed to parse questions: {e} | raw={qa_raw!r}")
            doc.metadata["answeredQuestions"] = []

        updated_docs.append(doc)
    
    return {"documents": updated_docs}

def create_questions_answered(state: State_buildIndex) -> dict:
    questions_answered: List[Document] = []

    documents = state["documents"]
    for doc in documents:
        questions = doc.metadata.get("answeredQuestions", []) or []
        for q in questions:
            new_doc = Document(text=q)
            new_doc.metadata["severity"] = doc.metadata.get("severity", "")
            new_doc.metadata["from_doc_id"] = getattr(doc, "doc_id", None)
            print(f'newdoc: {new_doc.text}, {new_doc.metadata}')
            questions_answered.append(new_doc)

    return {"questions_answered": questions_answered}

def create_mainkeywords_and_keywords(state: State_buildIndex) -> dict:
    msg = state["llm"].invoke(
        f'Based on the text I provide you, generate a list of relevant mainkeywords and keywords: {state["documents_text"]}\n\n'
        ' Output requirements:\n'
        '- Answer in norwegian\n'
        '- Return JSON ONLY. No explanations. No markdown. No code fences.\n'
        '- Use double quotes for all keys and strings.\n'
        '- Do not include trailing commas.\n'
        '- Avoid duplicate key\n'
        '- Example for a mainkeywords: "Følelser og tanker", "Forhold og Identitet"\n'
        '- Example for a keywords: "Forelskelse", "Forhold"\n'
        '- Shape:\n'
        '[{"Mainkeywords": "mainkeywords", "Keywords": ["keyword1", "keyword2"]}, ...]\n'
    )
    print(f'keywords raw: {msg.content}')

    try:
        keyword_sets = json.loads(msg.content)
        if not isinstance(keyword_sets, list):
            raise ValueError("keyword_sets must be a list")
    except Exception as e:
        # fail-soft: keep pipeline alive with empty list
        print(f'Failed to parse keyword_sets JSON: {e}')
        keyword_sets = []

    # IMPORTANT: return updates, don’t mutate state in-place
    return {"keyword_sets": keyword_sets}

def create_log_file(state: State_buildIndex) -> dict:
    # Create a log for each item
    # item = state["name"]
    # documents = state["documents"]
    # with open(f'combined_documents_for_{item}.txt', 'w', encoding='utf-8') as file:
    #     print('--------------------------')
    #     for doc in documents:
    #         # Robust printing for console
    #         print(f'doc.metadata: {doc.metadata}')
    #         print(f'doc.text: {(doc.text[:200] + "...") if isinstance(doc.text, str) and len(doc.text) > 200 else doc.text}')

    #         # Serialize metadata as JSON
    #         meta_json = json.dumps(doc.metadata or {}, ensure_ascii=False)
    #         file.write(meta_json + '\n')

    #         # Ensure text is a string
    #         text_str = doc.text if isinstance(doc.text, str) else ("" if doc.text is None else str(doc.text))
    #         file.write(text_str + '\n\n')
    import json
import pandas as pd
from pandas import json_normalize
from pathlib import Path

# def create_log_excel(state: dict) -> dict:
#     """
#     Writes two sheets:
#       - 'raw': metadata as JSON + full text
#       - 'flattened': metadata expanded into columns (best-effort)
#     File name: combined_documents_for_{state['name']}.xlsx
#     """
#     item = state.get("name", "item")
#     documents = state.get("documents", [])
#     safe_item = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(item))
#     out_path = Path(f"combined_documents_for_{safe_item}.xlsx")

#     rows = []
#     for idx, doc in enumerate(documents):
#         meta = getattr(doc, "metadata", None) or {}
#         text = getattr(doc, "text", None)
#         text_str = text if isinstance(text, str) else ("" if text is None else str(text))
#         rows.append({
#             "row_index": idx,
#             "item": item,
#             "metadata_json": json.dumps(meta, ensure_ascii=False),
#             "text": text_str,
#         })

#         # Optional: console trace (like your original)
#         print('--------------------------')
#         print(f'doc.metadata: {meta}')
#         preview = (text_str[:200] + "...") if isinstance(text_str, str) and len(text_str) > 200 else text_str
#         print(f'doc.text: {preview}')

#     # Build DataFrames
#     df_raw = pd.DataFrame(rows)

#     # Flatten metadata into columns (best-effort)
#     # If metadata contains nested dicts/lists, json_normalize helps expand keys with dotted paths.
#     meta_dicts = [json.loads(r["metadata_json"]) if r["metadata_json"] else {} for r in rows]
#     df_meta = json_normalize(meta_dicts, sep=".")
#     df_flat = pd.concat([df_raw.drop(columns=["metadata_json"]), df_meta], axis=1)

#     # Write to Excel with a stable engine
#     with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
#         # Sheet 1: raw
#         df_raw.to_excel(writer, sheet_name="raw", index=False)
#         # Sheet 2: flattened
#         df_flat.to_excel(writer, sheet_name="flattened", index=False)

#         # Basic formatting (optional)
#         workbook  = writer.book
#         wrap_fmt  = workbook.add_format({"text_wrap": True, "valign": "top"})
#         for sheet, text_col in (("raw", "text"), ("flattened", "text")):
#             ws = writer.sheets[sheet]
#             # Auto-ish width for first few columns
#             for col_idx, col_name in enumerate(df_raw.columns if sheet=="raw" else df_flat.columns):
#                 # Wider column for text
#                 width = 80 if col_name == text_col else 24
#                 ws.set_column(col_idx, col_idx, width, wrap_fmt if col_name == text_col else None)

#     print(f"Excel written to: {out_path.resolve()}")
#     return {}

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
        print('--------------------------')
        print(f'doc.metadata: {meta}')
        preview = (text_str[:200] + "...") if isinstance(text_str, str) and len(text_str) > 200 else text_str
        print(f'doc.text: {preview}')

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
    qa_docs   = state.get("questions_answered", [])
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
builder.add_node("create_mainkeywords_and_keywords", create_mainkeywords_and_keywords)
builder.add_node("create_metadata_for_documents", create_metadata_for_documents)
builder.add_node("create_questionsAnswered_bank", create_questionsAnswered_bank)

builder.add_node("create_log_excel", create_log_excel)

builder.add_edge(START, "apify_call_load_documents")
builder.add_edge("apify_call_load_documents", "create_metadata_for_documents")
builder.add_edge("create_mainkeywords_and_keywords", "create_metadata_for_documents")

builder.add_edge("create_metadata_for_documents", "create_questions_answered")
builder.add_edge("create_questions_answered", "create_log_excel")
#builder.add_edge("apify_call_load_documents", "create_key_words")
builder.add_edge("create_log_excel", "persist_storage")

# 5️⃣ Finally, aggregator → END

builder.add_edge("persist_storage", END)

build_hei_index_workflow = builder.compile()

logging.info("answer_witth_related_queries_workflow created...")