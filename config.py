# config.py

import os
import logging
import time
import json
from llama_index.core import (StorageContext, load_index_from_storage)
from collections import namedtuple
import asyncio

from dotenv import load_dotenv, find_dotenv

from llm_provider import build_chat_llm, build_fast_chat_llm
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
        self.fast_llm = None

    def update_status(self, status):
        self.status = status
        if status == "Server is ready":
            self.indexes_loaded = True

    def set_llm(self, llm):
        self.llm = llm

    def set_fast_llm(self, llm):
        self.fast_llm = llm

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
server_settings.set_fast_llm(build_fast_chat_llm())



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

        # Precompute the /examples category pools so the first request is fast.
        # Local import avoids a circular import (answer_utils imports config).
        if loaded:
            try:
                from answer_utils import warm_examples_cache
                await warm_examples_cache(vector_store)
            except Exception:
                logging.exception("Failed to warm examples cache at startup")

    except Exception as e:
        logging.error(f"Failed to read indexes from storage: {e}")
        server_settings.update_status("Server is not ready")
        vector_store.indexes_loaded = False


def check_index_consistency(name, idx):
    """Verify the vector store and docstore of a loaded index agree.

    The fatal condition is an *orphan embedding*: a vector in the store whose
    node ID is missing from the docstore. When such a node lands in a query's
    top-k, LlamaIndex raises "Node ID ... not found in fetched nodes" and the
    whole answer is rejected. (Docstore entries without an embedding are only
    informational — ref-docs legitimately have none, and they just can't be
    retrieved.)

    Logs the result and returns the number of orphan embeddings
    (0 = consistent, -1 = check could not run).
    """
    try:
        vstore = idx.vector_store
        emb = getattr(getattr(vstore, "data", None), "embedding_dict", None)
        if emb is None:
            emb = getattr(getattr(vstore, "_data", None), "embedding_dict", None)
        if emb is None:
            logging.warning(
                "Index '%s': could not read embedding_dict (vector store type %s) — skipping consistency check.",
                name, type(vstore).__name__,
            )
            return -1

        emb_ids = set(emb.keys())
        doc_ids = set(idx.docstore.docs.keys())
        orphans = emb_ids - doc_ids
        no_emb = doc_ids - emb_ids

        if orphans:
            logging.error(
                "Index '%s' INCONSISTENT: %d of %d embeddings reference node IDs missing from the "
                "docstore — retrieval will crash whenever one is in top-k. %d docstore entries have no "
                "embedding. Rebuild the index. Sample orphan IDs: %s",
                name, len(orphans), len(emb_ids), len(no_emb), list(orphans)[:3],
            )
        else:
            logging.info(
                "Index '%s' consistency OK: %d embeddings, all present in docstore "
                "(%d docstore entries, %d without embedding).",
                name, len(emb_ids), len(doc_ids), len(no_emb),
            )
        return len(orphans)
    except Exception:
        logging.exception("Could not run consistency check for index '%s'", name)
        return -1


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
            # Flag a vector-store/docstore mismatch right at load (e.g. after a
            # rebuild) so a corrupt index surfaces in the startup log.
            check_index_consistency(name, idx)
            found_any = True
        else:
            logging.warning(f"Index directory not found: {storage}")

        elapsed = time.time() - start
        logging.info(f"Time taken for {name}: {elapsed:.2f}s")

    return found_any