"""
OpenAI chat LLM client.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openai import OpenAI

from app.config import settings

DEFAULT_LLM_MODEL = settings.llm_model_name
DEFAULT_TEMPERATURE = 0.0


class LLMClient:
    def __init__(
        self,
        model: str = DEFAULT_LLM_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        client: OpenAI | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        self.client = client or OpenAI(api_key=api_key)

    def chat(self, messages: List[Dict[str, Any]], response_format: Optional[Dict[str, Any]] = None) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0].message
        return choice.content or ""


__all__ = ["LLMClient", "DEFAULT_LLM_MODEL", "DEFAULT_TEMPERATURE"]

