# config.py

import os
import logging
import time
import json
from llama_index.core import (StorageContext, load_index_from_storage)
from collections import namedtuple
import asyncio

from dotenv import load_dotenv, find_dotenv

from llm_provider import build_chat_llm
from embeddings_provider import configure_embeddings

load_dotenv(find_dotenv(), override=True)

# define the namedtuple at module scope
IndexObject = namedtuple('IndexObject', ['name', 'index', 'description'])


def RunningLocally():
    if 'WEBSITE_SITE_NAME' in os.environ or 'FUNCTIONS_WORKER_RUNTIME' in os.environ:
        return False
    else:
        print("Logging info locally")
        return True
    
# Class definitions
class CustomError(Exception):
    def __init__(self, message, code):
        super().__init__(message)
        self.code = code

class ServerSettings:
    def __init__(self):
        self.indexes_loaded = False
        self.status = "Server is not ready"
        self.llm = None

    def update_status(self, status):
        self.status = status
        if status == "Server is ready":
            self.indexes_loaded = True

    def set_llm(self, llm):
        self.llm = llm

    def get_status(self):
        return self.status, self.indexes_loaded
    
    def __str__(self):
        # Convert object properties to a JSON string
        return json.dumps(self.__dict__, ensure_ascii=False, indent=4, default=str)

configure_embeddings()

server_settings = ServerSettings()


class VectorIndexStore:
    """Singleton store for all loaded vector indexes."""
    def __init__(self):
        self.indexes_loaded = False
        self.objects: list[IndexObject] = []

    def add(self, name, index_obj, description):
        """Append a new IndexObject."""
        self.objects.append(IndexObject(name, index_obj, description))

    def get(self, name):
        """
        Retrieve the IndexObject by name.
        Returns the namedtuple or None if not found.
        """
        for entry in self.objects:
            if entry.name == name:
                return entry
        return None

    def clear(self):
        """Clear all stored indexes."""
        self.objects.clear()

    def get_all(self):
        """Return a list of all stored entries."""
        return list(self.objects)
    
    def __str__(self):
        # Convert object properties to a JSON string
        return json.dumps(self.__dict__, ensure_ascii=False, indent=4, default=str)


# instantiate the singleton
vector_store = VectorIndexStore()


VECTOR_INDEX_MAP = [
    {"name": "hvaerinnafor", "storage": ("." if RunningLocally() else "") +"/blobstorage/chatbot/hvaerinnafor", "description":"Forelskelse"},
    {"name": "hvaerinnafor_qa_bank", "storage": ("." if RunningLocally() else "") +"/blobstorage/chatbot/hvaerinnafor_qa_bank", "description":"Relaterte spørsmål"},
    {"name": "hvaerinnafor_unified", "storage": ("." if RunningLocally() else "") +"/blobstorage/chatbot/hvaerinnafor_unified", "description":"hvaerinnafor_unified is a single vector index that merges two content types: article chunks from the hvaerinnafor knowledge base, and ~7900 real Q&A entries from ung.no (last 12 months). Every node — regardless of source — exposes an answer-text field that the LLM sees, plus an embedding-only aliases field of question phrasings that shapes retrieval without leaking into the LLM prompt. For articles, aliases are 10 LLM-generated synthetic questions per chunk; for Q&A nodes, the original user question. At build time, real ung.no questions are additionally cross-pollinated onto the article chunks they best match (top-3, similarity ≥ 0.55, score-ranked). At query time, articles and Q&A compete in the same retrieval call — a chunk wins whether the user's wording resembles its raw text, a question its author imagined, or a question someone has actually asked."}
]


server_settings.set_llm(build_chat_llm())



def init_env_and_logging():
    logging.basicConfig(
        level=logging.INFO if RunningLocally() else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True
    )

    # Silence noisy HTTP request logs from the OpenAI/Azure stack
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Sometimes these also produce noise depending on versions
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("azure").setLevel(logging.WARNING)

async def async_read_indexes():
    logging.info("Starting to read indexes...")
    status, _ = server_settings.get_status()
    logging.info(f"Current Server Status: {status}")

    try:
        # clear any previous run
        vector_store.clear()
        
        # offload the sync work
        found_any = await asyncio.to_thread(read_all_indexes_from_storage, VECTOR_INDEX_MAP)


        if found_any:
            logging.info("Indexes successfully read from storage.")
            server_settings.update_status("Server is ready")
        else:
            logging.info("Indexes not successfully read from storage.")
            server_settings.update_status("Server is not ready")

        status, loaded = server_settings.get_status()
        vector_store.indexes_loaded = loaded
        logging.info(f"Updated Server Status: {status}")

    except Exception as e:
        logging.error(f"Failed to read indexes from storage: {e}")
        server_settings.update_status("Server is not ready")
        vector_store.indexes_loaded = False


def read_all_indexes_from_storage(vector_map):
    """Load all indexes into the singleton store."""
    found_any = False

    for item in vector_map:
        start = time.time()
        name = item['name']
        storage = item['storage']
        desc = item['description']
        logging.info("-------------------------------")

        if os.path.exists(storage):
            logging.info(f"Loading index '{name}' from {storage}")
            storage_ctx = StorageContext.from_defaults(persist_dir=storage)
            idx = load_index_from_storage(storage_ctx)
            # correctly add to the store
            vector_store.add(name, idx, desc)
            found_any = True
        else:
            logging.warning(f"Index directory not found: {storage}")

        elapsed = time.time() - start
        logging.info(f"Time taken for {name}: {elapsed:.2f}s")

    return found_any