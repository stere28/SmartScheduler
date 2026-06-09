"""
SmartScheduler – LLM Provider Factory
======================================
Centralises LLM instantiation so that switching between providers
(Ollama, OpenAI, …) requires only an environment-variable change,
with zero modifications to agent code.

Environment variables:
  LLM_PROVIDER      "ollama" (default) | "openai"
  LLM_MODEL         model name         (default depends on provider)
  LLM_TEMPERATURE   float              (default 0.2)
  OLLAMA_BASE_URL   Ollama server URL  (default "http://localhost:11434")
  OPENAI_API_KEY    required only when LLM_PROVIDER=openai
"""

from __future__ import annotations

import os
from langchain_core.language_models.chat_models import BaseChatModel


# ── Supported providers ────────────────────────────────────────────────────────
_PROVIDER_OLLAMA = "ollama"
_PROVIDER_OPENAI = "openai"

_DEFAULT_MODELS: dict[str, str] = {
    _PROVIDER_OLLAMA: "llama3.2",
    _PROVIDER_OPENAI: "gpt-4o",
}


def get_llm() -> BaseChatModel:
    """
    Return a configured LangChain chat model based on environment variables.

    Returns
    -------
    BaseChatModel
        A ready-to-use LangChain chat model instance.

    Raises
    ------
    ValueError
        If the requested provider is not supported or required configuration
        is missing.
    """
    provider    = os.environ.get("LLM_PROVIDER", _PROVIDER_OLLAMA).lower().strip()
    temperature = float(os.environ.get("LLM_TEMPERATURE", "0.2"))

    if provider == _PROVIDER_OLLAMA:
        return _build_ollama(temperature)
    elif provider == _PROVIDER_OPENAI:
        return _build_openai(temperature)
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER='{provider}'. "
            f"Supported values: {[_PROVIDER_OLLAMA, _PROVIDER_OPENAI]}"
        )


# ── Private builders ───────────────────────────────────────────────────────────

def _build_ollama(temperature: float) -> BaseChatModel:
    """Instantiate a ChatOllama model."""
    try:
        from langchain_ollama import ChatOllama  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "langchain-ollama is not installed. "
            "Run: pip install langchain-ollama"
        ) from exc

    model       = os.environ.get("LLM_MODEL", _DEFAULT_MODELS[_PROVIDER_OLLAMA])
    base_url    = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    print(f"  [LLM] Provider: Ollama | Model: {model} | URL: {base_url}")
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        format="json",      # Force JSON output mode – avoids empty/non-JSON replies
        num_ctx=4096,       # Ensure long prompts are not truncated
    )


def _build_openai(temperature: float) -> BaseChatModel:
    """Instantiate a ChatOpenAI model."""
    try:
        from langchain_openai import ChatOpenAI  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "langchain-openai is not installed. "
            "Run: pip install langchain-openai"
        ) from exc

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "LLM_PROVIDER=openai requires OPENAI_API_KEY to be set."
        )

    model = os.environ.get("LLM_MODEL", _DEFAULT_MODELS[_PROVIDER_OPENAI])

    print(f"  [LLM] Provider: OpenAI | Model: {model}")
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
    )
