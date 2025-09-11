import logging
import os
import time
from llama_index.core import (VectorStoreIndex, StorageContext, Settings,  load_index_from_storage)
import llama_index.core.readers as readers
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader


def download_and_persist_storage(name, storage):
    logging.info('1 - download_and_persist_storage')
    documents = download_documents_for_item(name)
    print (f'2 - Loaded {len(documents)} documents')

    # Split documents into Node objectts
    #nodes = SentenceSplitter.from_defaults(chunk_size=256, chunk_overlap=75).get_nodes_from_documents(documents)
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

    if ('WEBSITE_SITE_NAME' in os.environ or 'FUNCTIONS_WORKER_RUNTIME' in os.environ):
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


    else:
        # running local

        # persist locally only (you need to copy the files manually to the blobcontainer)
        storage_context.persist(persist_dir=storage)


    logging.info('7 - storage_context.persist ok')

    return index