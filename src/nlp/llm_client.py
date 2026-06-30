import os
from dataclasses import dataclass
from typing import Protocol

from dotenv import load_dotenv
from ollama import Client, ResponseError


class LLMClient(Protocol):
    """
    Generic interface for LLM backends.

    Evidence extraction, report generation, etc. should depend on this interface,
    not on a specific provider like Ollama Cloud, local Ollama, or another hosted API.
    """

    def generate_json(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
    ) -> str:
        ...


@dataclass
class OllamaChatClient:
    """
    Ollama client using the official ollama Python package.

    This can work with either:
    - Ollama Cloud: OLLAMA_BASE_URL=https://ollama.com
    - Local Ollama: OLLAMA_BASE_URL=http://localhost:11434
    
    """

    base_url: str
    api_key: str | None = None

    @classmethod
    def from_env(cls) -> "OllamaChatClient":
        load_dotenv()

        base_url = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
        api_key = os.getenv("OLLAMA_API_KEY")

        return cls(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
        )

    def _build_client(self) -> Client:
        headers: dict[str, str] = {}

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return Client(
            host=self.base_url,
            headers=headers,
        )

    def generate_json(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
    ) -> str:
        client = self._build_client()

        try:
            response = client.chat(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                format="json",
                stream=False,
                options={
                    "temperature": temperature,
                },
            )
        except ResponseError as exc:
            raise RuntimeError(f"Ollama API error: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        try:
            return response["message"]["content"]
        except KeyError as exc:
            raise RuntimeError(f"Ollama response missing message content: {response}") from exc


def get_default_llm_client() -> LLMClient:
    """
    Factory function for the default LLM backend.

    Later, if we add a different backend, this is where we can switch based on
    an environment variable like LLM_PROVIDER.
    """
    return OllamaChatClient.from_env()