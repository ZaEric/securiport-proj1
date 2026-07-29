import os
from dataclasses import dataclass
from typing import Any
from typing import Protocol

# logging, trying to make outputs completely deterministic (might be impossible with Ollama, but gonna test it anyways)
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False

try:
    from ollama import Client, ResponseError
except ImportError:
    Client = None  # type: ignore[assignment]

    class ResponseError(Exception):
        pass


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
        seed: int | None = 42,
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

    def _build_client(self) -> Any:
        if Client is None:
            raise RuntimeError(
                "The ollama package is not installed. "
                "Install project dependencies before running LLM endpoints."
            )

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
        seed: int | None = 42,
    ) -> str:
        client = self._build_client()

        try:
            # options: dict[str, float | int] = {
            #     "temperature": temperature,
            #     "top_k": 1,
            #     "top_p": 1.0,
            #     "num_ctx": 8192,
            #     "num_thread": 1,
            # }

            # if seed is not None:
            #     options["seed"] = seed

            # hardcode temp and seed just to make sure
            options = {
                "temperature": temperature,
                "seed": seed,
                # "num_ctx": 8192,
                "top_k": 1, # top_k is supposed to enforce greedy decoding which should make outputs deterministic, but it seem to be working
                # "top_p": 1.0,
            }

            # keep_alive = "30m" # unload model after every request, see if it helps make outputs deterministic (at cost of increased runtime)

            response = client.generate(
                model=model,
                prompt=prompt,
                format="json",
                stream=False,
                options=options,
            )

            write_ollama_debug_log(
                model=model,
                prompt=prompt,
                response=response,
                options=options,
                keep_alive=None,
            )
        
        except ResponseError as exc:
            raise RuntimeError(f"Ollama API error: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        try:
            return response["response"]
        except KeyError as exc:
            raise RuntimeError(f"Ollama response missing response content: {response}") from exc


def write_ollama_debug_log(
    *,
    model: str,
    prompt: str,
    response: dict[str, Any],
    options: dict[str, Any],
    keep_alive: Any,
) -> None:
    """
    Appends Ollama request/response metadata to data/logs/ollama_requests.jsonl.

    Does not log the full prompt or full response text by default.
    Instead, logs hashes so repeated identical prompts/outputs can be compared.
    """
    logs_dir = Path("data/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    response_text = response.get("response", "")

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_requested": model,
        "model_returned": response.get("model"),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "response_sha256": hashlib.sha256(str(response_text).encode("utf-8")).hexdigest(),
        "options": options,
        "keep_alive": keep_alive,
        "created_at": response.get("created_at"),
        "done": response.get("done"),
        "done_reason": response.get("done_reason"),
        "total_duration": response.get("total_duration"),
        "load_duration": response.get("load_duration"),
        "prompt_eval_count": response.get("prompt_eval_count"),
        "prompt_eval_duration": response.get("prompt_eval_duration"),
        "eval_count": response.get("eval_count"),
        "eval_duration": response.get("eval_duration"),
    }

    log_path = logs_dir / "ollama_requests.jsonl"

    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def get_default_llm_client() -> LLMClient:
    """
    Factory function for the default LLM backend.

    Later, if we add a different backend, this is where we can switch based on
    an environment variable like LLM_PROVIDER.
    """
    return OllamaChatClient.from_env()
