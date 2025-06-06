# config.py

import os
import logging
import time
import json
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from llama_index.core import (StorageContext, load_index_from_storage)
from collections import namedtuple
import asyncio

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
    {"name": "ungnotobakk", "storage": ("." if RunningLocally() else "") +"/blobstorage/chatbot/ungnotobakk", "description":"Jeg svarer på spørsmål om tobakk, snus, vaping"},
    {"name": "ungnospmtobakk", "storage": ("." if RunningLocally() else "") +"/blobstorage/chatbot/ungnospmtobakk", "description":"Jeg svarer på spørsmål om tobakk, snus, vaping"}, 
    {"name": "ungnospm", "storage": ("." if RunningLocally() else "") +"/blobstorage/chatbot/ungnospm", "description":"Jeg svarer på spørsmål ulike temaer som ungdommer lurer på"},     
    {"name": "helsenorgeartikler", "storage": ("." if RunningLocally() else "") +"/blobstorage/chatbot/helsenorgeartikler", "description":"Jeg svarer på spørsmål om flere artikler på helsenorge om helsespørsmål som graviditet, rus, tobakk, graviditet og sykdommer"},
]


LLMGPT4 = AzureChatOpenAI(
    model=os.getenv('AZURE_OPENAI_MODEL'),
    deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
    azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    azure_endpoint=os.getenv('AZURE_OPENAI_AZURE_ENDPOINT'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSJON'),
    temperature=0.0,
    timeout=120,
)

server_settings.set_llm(LLMGPT4)



def init_env_and_logging():
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO if RunningLocally() else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True
    )


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
