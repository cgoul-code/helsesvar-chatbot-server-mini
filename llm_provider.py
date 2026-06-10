"""Chat-LLM factory.

Selects the chat model via the LLM_PROVIDER env var so the rest of the codebase
can stay provider-agnostic. The workflow only relies on LangChain's BaseChatModel
interface (invoke, with_structured_output, usage_metadata callbacks), which both
returned objects implement.
"""

import os

from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI
from langchain_anthropic import ChatAnthropic


def build_chat_llm() -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "azure_openai").lower()

    if provider == "azure_openai":
        return AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            timeout=120,
            temperature=0.0,
            verbose=False,
        )

    if provider == "anthropic":
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            timeout=120,
            temperature=0.0,
            max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS", "4096")),
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider!r}. "
        "Expected one of: 'azure_openai', 'anthropic'."
    )


def build_fast_chat_llm() -> BaseChatModel:
    """A cheaper/faster model for the auxiliary calls (query analysis,
    orchestration, entailment gate, related-query selection).

    Configured via *_FAST_* env vars. If none is set we fall back to the
    main model, so behaviour is unchanged until a fast deployment exists.
    """
    provider = os.getenv("LLM_PROVIDER", "azure_openai").lower()

    if provider == "azure_openai":
        fast_deployment = os.getenv("AZURE_OPENAI_FAST_DEPLOYMENT_NAME")
        if not fast_deployment:
            return build_chat_llm()
        return AzureChatOpenAI(
            azure_deployment=fast_deployment,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            timeout=60,
            temperature=0.0,
            verbose=False,
        )

    if provider == "anthropic":
        fast_model = os.getenv("ANTHROPIC_FAST_MODEL")
        if not fast_model:
            return build_chat_llm()
        return ChatAnthropic(
            model=fast_model,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            timeout=60,
            temperature=0.0,
            max_tokens=int(os.getenv("ANTHROPIC_FAST_MAX_TOKENS", "2048")),
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider!r}. "
        "Expected one of: 'azure_openai', 'anthropic'."
    )
