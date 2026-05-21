"""Embeddings factory.

Kept separate from llm_provider.py because the chat-LLM and the embeddings
backend are independent choices. Anthropic doesn't offer an embeddings API,
so when LLM_PROVIDER=anthropic the embeddings can still come from Azure
OpenAI (or OpenAI direct, Voyage, etc.) without any change to the workflow.
"""

import os

from llama_index.core import Settings
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding


def configure_embeddings() -> None:
    provider = os.getenv("EMBEDDINGS_PROVIDER", "azure_openai").lower()

    if provider == "azure_openai":
        Settings.embed_model = AzureOpenAIEmbedding(
            model=os.getenv("AZURE_OPENAI_EMBEDDINGS_MODEL"),
            deployment_name=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
            api_key=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_VERSION"),
        )
        return

    raise ValueError(
        f"Unknown EMBEDDINGS_PROVIDER: {provider!r}. "
        "Currently supported: 'azure_openai'."
    )
