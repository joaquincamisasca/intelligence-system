"""
Fábrica de chat models de LangChain (mismo patrón de proveedor
intercambiable que los módulos 1 y 2): el resto del sistema pide un LLM
con `get_llm()` y no sabe si detrás hay Ollama, OpenAI o Anthropic.

Todos estos modelos exponen la interfaz async (`ainvoke`, `astream`) y
usan los clientes asíncronos oficiales de cada proveedor, así que se
pueden esperar con `await` sin bloquear el event loop.
"""

from typing import Optional

from app.config import settings


def get_llm(provider: Optional[str] = None, model: Optional[str] = None, temperature: float = 0.0):
    provider = provider or settings.generation_provider

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        clave = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        return ChatOpenAI(model=model or settings.openai_model, temperature=temperature, api_key=clave)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        clave = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
        return ChatAnthropic(model=model or settings.anthropic_model, temperature=temperature, api_key=clave)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model or settings.ollama_model,
            temperature=temperature,
            base_url=settings.ollama_base_url,
            num_ctx=settings.ollama_num_ctx,
        )

    raise ValueError(f"Proveedor no soportado: {provider}")
