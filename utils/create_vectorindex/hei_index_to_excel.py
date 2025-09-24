
import os
import logging
import json
import time
from collections import namedtuple

from dotenv import load_dotenv, find_dotenv
from llama_index.core import StorageContext, load_index_from_storage
import pandas as pd
from pathlib import Path
from llama_index.core import VectorStoreIndex



# ==== Index store ====
IndexObject = namedtuple('IndexObject', ['name', 'index', 'description'])

class VectorIndexStore:
    """Singleton-like store for all loaded vector indexes."""
    def __init__(self):
        self.indexes_loaded = False
        self.objects: list[IndexObject] = []

    def add(self, name, index_obj, description):
        self.objects.append(IndexObject(name, index_obj, description))

    def get(self, name):
        for entry in self.objects:
            if entry.name == name:
                return entry
        return None

    def clear(self):
        self.objects.clear()

    def get_all(self):
        return list(self.objects)

    def __str__(self):
        return json.dumps(self.__dict__, ensure_ascii=False, indent=4, default=str)
    

def RunningLocally():
    if 'WEBSITE_SITE_NAME' in os.environ or 'FUNCTIONS_WORKER_RUNTIME' in os.environ:
        return False
    else:
        print("Logging info locally")
        return True
VECTOR_INDEX_MAP = [
    {
        "name": "hvaerinnafor",
        "storage": ("." if RunningLocally() else "") + "/blobstorage/chatbot/hvaerinnafor",
        "description": "Forelskelse",
    },
    {
        "name": "hvaerinnafor_qa_bank",
        "storage": ("." if RunningLocally() else "") + "/blobstorage/chatbot/hvaerinnafor_qa_bank",
        "description": "Relaterte spørsmål",
    },
]

# --- Put this helper in your module ---
def _node_text(node) -> str:
    # LlamaIndex nodes usually have get_text(); fallback to .text
    if hasattr(node, "get_text") and callable(node.get_text):
        return node.get_text()
    return getattr(node, "text", "")

def _node_ref_doc_id(node) -> str:
    # Different versions use different attrs
    return getattr(node, "ref_doc_id", "") or getattr(node, "doc_id", "")

def export_indexes_to_excel(vector_store, xlsx_path: str):
    """
    Export all nodes (text + metadata) from each loaded index in vector_store
    into a single Excel file, one sheet per index name.
    """
    if not vector_store.get_all():
        raise RuntimeError("No indexes loaded in vector_store.")

    # Ensure parent dir exists
    Path(xlsx_path).parent.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        for entry in vector_store.get_all():
            idx = entry.index
            name = entry.name

            # Access docstore from the index's storage_context
            try:
                docstore = idx.storage_context.docstore
            except AttributeError as e:
                raise RuntimeError(
                    f"Index '{name}' does not expose storage_context.docstore; "
                    "ensure you're on a recent LlamaIndex version or keep the StorageContext when loading."
                ) from e

            # get_all_documents() returns a dict of {node_id: BaseNode}
            nodes = docstore.docs.items()
            print(f'Antal noder:{len(nodes)}')
            rows = []
            # Collect *all* metadata keys to keep columns consistent
            all_meta_keys = set()

            # First pass: build rows and gather metadata keys
            temp = []
            docstore = idx.storage_context.docstore
            nodes = docstore.docs
            docs = {}
            
            for doc_id, node in nodes.items():
                meta = getattr(node, "metadata", None) or {}
                text = getattr(node, "text", None)
                if name == "hvaerinnafor":
                    print(f'doc_id:{doc_id}, meta={meta}')

                # Excel max cell limit ~32,767 chars; you can truncate if desired:
                # text = text[:32760]

                row = {
                    "node_id": getattr(node, "node_id", ""),
                    "ref_doc_id": _node_ref_doc_id(node),
                    "text": text,
                }
                temp.append((row, meta))
                all_meta_keys.update(map(str, meta.keys()))
                

            # Second pass: flatten metadata with stable columns
            cols_meta = [f"meta.{k}" for k in sorted(all_meta_keys)]
            final_rows = []
            for row, meta in temp:
                flat = row.copy()
                # Fill each known meta column; missing -> empty
                for k in sorted(all_meta_keys):
                    flat[f"meta.{k}"] = meta.get(k, "")
                final_rows.append(flat)

            df = pd.DataFrame(final_rows, columns=["node_id", "ref_doc_id", "text"] + cols_meta)
            # Sheet names must be <=31 chars and unique
            safe_sheet = name[:31] or "index"
            df.to_excel(writer, sheet_name=safe_sheet, index=False)

    logging.info("Export complete: %s", xlsx_path)

    

# module-level store
vector_store = VectorIndexStore()

def read_all_indexes_from_storage(vector_map):
    """Load all indexes into the module-level vector_store."""
    global vector_store
    found_any = False

    # Do NOT re-instantiate vector_store here; we fill the existing one.
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
            vector_store.add(name, idx, desc)
            found_any = True
        else:
            logging.warning(f"Index directory not found: {storage}")

        elapsed = time.time() - start
        logging.info(f"Time taken for {name}: {elapsed:.2f}s")

    vector_store.indexes_loaded = found_any
    return found_any

logging.basicConfig(
    level=logging.INFO,
    force=True
)

# ---------------------------------------



load_dotenv(find_dotenv())

# Load indexes
try:
    found_any = read_all_indexes_from_storage(VECTOR_INDEX_MAP)
    if found_any:
        logging.info("Indexes successfully read from storage.")
    else:
        logging.info("Indexes not successfully read from storage.")
except Exception as e:
    logging.error(f"Failed to read indexes from storage: {e}")
    vector_store.indexes_loaded = False

# --- Use the store ---
# NOTE: correct the name here ("hvaerinnafor", not "hvaerinnfor")
entry = vector_store.get("hvaerinnafor")
if entry is None:
    raise RuntimeError("Index 'hvaerinnafor' not found. Loaded: "
                       + ", ".join([e.name for e in vector_store.get_all()]))

index = entry.index  # type: ignore
vector_index_description = entry.description
logging.info("Found entry: %s", vector_index_description)

export_indexes_to_excel(vector_store, "./exports/vector_indexes.xlsx")