from typing import Any, Dict, List, Type, Optional

import ollama
from pydantic import BaseModel, ValidationError
import logging
import json

from config.settings import get_settings


class LLMFactory:
    """Factory class to initialize different LLM providers dynamically (only supports DeepSeek via Ollama)."""

    def __init__(self):
        self.settings = get_settings().deepseek

        if not self.settings:
            raise ValueError("❌ DeepSeek configuration is missing in settings.")

        self.client = self._initialize_client()

    def _initialize_client(self) -> Any:
        """Initialize the client for DeepSeek via Ollama."""
        return {"model": self.settings.default_model}  # Ollama ne nécessite pas d'API Key

    def create_completion(
            self, response_model: Optional[Type[BaseModel]], messages: List[Dict[str, str]], **kwargs
    ) -> Any:
        """Generate a completion response using DeepSeek via Ollama.

        Args:
            response_model: A Pydantic model to parse the response (optional).
            messages: List of chat messages.

        Returns:
            Parsed response if response_model is provided, otherwise raw text.
        """

        response = ollama.chat(
            model=self.client["model"],
            messages=messages,
            options={
                "temperature": kwargs.get("temperature", self.settings.temperature),
                "num_ctx": kwargs.get("max_tokens", self.settings.max_tokens),
            },
        )

        # Extraire la réponse texte brute
        print(response)
        if "message" in response:
            raw_text = response["message"]["content"]
        else:
            logging.error(f"❌ Error from Ollama: {response}")
            return None

        logging.info(f"📝 Ollama Response: {raw_text}")

        # 🛠️ Tentative de parsing en JSON si un response_model est fourni
        if response_model:
            try:
                json_response = json.loads(raw_text)  # Convertit la réponse en JSON
                parsed_response = response_model.parse_obj(json_response)  # Vérifie la structure
                return parsed_response
            except (json.JSONDecodeError, ValidationError) as e:
                logging.error(f"❌ Error parsing response: {e}")
                return None

        return raw_text  # Retourne du texte brut si pas de `response_model`
