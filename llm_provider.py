"""Chat-LLM factory.

Selects the chat model via the LLM_PROVIDER env var so the rest of the codebase
can stay provider-agnostic. The workflow only relies on LangChain's BaseChatModel
interface (invoke, with_structured_output, usage_metadata callbacks), which both
returned objects implement.
"""

import os

from langchain_core.language_models import BaseChatModel

# NB: provider-SDK-ene importeres LAT inne i hver gren, ikke på toppnivå.
# Slik krever det å importere denne modulen bare SDK-en for den valgte
# LLM_PROVIDER – en bruker på Azure OpenAI trenger ikke ha langchain_anthropic
# eller langchain_mistralai installert (og motsatt).


def build_chat_llm() -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "azure_openai").lower()

    if provider == "azure_openai":
        from langchain_openai import AzureChatOpenAI
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
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            timeout=120,
            temperature=0.0,
            max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS", "4096")),
        )

    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(
            model=os.getenv("MISTRAL_MODEL", "mistral-large-latest"),
            api_key=os.getenv("MISTRAL_API_KEY"),
            endpoint=os.getenv("MISTRAL_ENDPOINT") or None,
            timeout=120,
            temperature=0.0,
            max_tokens=int(os.getenv("MISTRAL_MAX_TOKENS", "4096")),
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider!r}. "
        "Expected one of: 'azure_openai', 'anthropic', 'mistral'."
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
        from langchain_openai import AzureChatOpenAI
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
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=fast_model,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            timeout=60,
            temperature=0.0,
            max_tokens=int(os.getenv("ANTHROPIC_FAST_MAX_TOKENS", "2048")),
        )

    if provider == "mistral":
        fast_model = os.getenv("MISTRAL_FAST_MODEL")
        if not fast_model:
            return build_chat_llm()
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(
            model=fast_model,
            api_key=os.getenv("MISTRAL_API_KEY"),
            endpoint=os.getenv("MISTRAL_ENDPOINT") or None,
            timeout=60,
            temperature=0.0,
            max_tokens=int(os.getenv("MISTRAL_FAST_MAX_TOKENS", "2048")),
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider!r}. "
        "Expected one of: 'azure_openai', 'anthropic', 'mistral'."
    )
