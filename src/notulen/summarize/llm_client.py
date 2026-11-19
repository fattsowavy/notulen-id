"""Chat clients for Ollama and OpenAI-compatible APIs."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any

import httpx

from notulen.config import LlmConfig, Secrets

logger = logging.getLogger(__name__)

Message = dict[str, str]


class LlmError(RuntimeError):
    """Raised when the LLM backend cannot produce a response."""


class _RetryableError(Exception):
    """A failure that may succeed if the request is sent again."""


class LlmClient(ABC):
    """Send chat requests with retries and clear error messages."""

    def __init__(
        self,
        config: LlmConfig,
        transport: httpx.BaseTransport | None = None,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        self.config = config
        self.retry_backoff_seconds = retry_backoff_seconds
        self._http = httpx.Client(timeout=config.timeout_seconds, transport=transport)

    def __enter__(self) -> LlmClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    def complete(self, prompt: str, system: str | None = None) -> str:
        """Return the model's reply to a single user prompt."""
        messages: list[Message] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        attempts = self.config.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                started = time.perf_counter()
                reply = self._send(messages)
            except _RetryableError as exc:
                if attempt == attempts:
                    raise LlmError(f"{exc} (after {attempts} attempt(s))") from exc
                delay = self.retry_backoff_seconds * 2 ** (attempt - 1)
                logger.warning("LLM request failed: %s. Retrying in %.1f s", exc, delay)
                time.sleep(delay)
                continue

            logger.info(
                "LLM %s replied in %.1f s", self.config.model, time.perf_counter() - started
            )
            if not reply.strip():
                raise LlmError(f"Model {self.config.model} returned an empty response")
            return reply
        raise AssertionError("unreachable")

    @abstractmethod
    def _send(self, messages: list[Message]) -> str:
        """Send one request and return the reply text."""

    @abstractmethod
    def _connection_hint(self) -> str:
        """Explain how to fix an unreachable server."""

    def _error_message(self, response: httpx.Response) -> str:
        """Describe a non-retryable HTTP error."""
        return f"LLM request failed with HTTP {response.status_code}: {response.text[:300]}"

    def _post(
        self, url: str, payload: dict[str, Any], headers: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """POST JSON and return the decoded body, classifying failures."""
        try:
            response = self._http.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise _RetryableError(
                f"Request to {url} timed out after {self.config.timeout_seconds:.0f} s"
            ) from exc
        except httpx.TransportError as exc:
            raise _RetryableError(f"{self._connection_hint()} ({exc})") from exc

        if response.status_code == 429 or response.status_code >= 500:
            raise _RetryableError(f"HTTP {response.status_code}: {response.text[:300]}")
        if response.is_error:
            raise LlmError(self._error_message(response))
        try:
            body: dict[str, Any] = response.json()
        except ValueError as exc:
            raise LlmError(f"LLM returned invalid JSON: {response.text[:300]}") from exc
        return body


class OllamaClient(LlmClient):
    """Client for the Ollama chat API."""

    def _send(self, messages: list[Message]) -> str:
        body = self._post(
            f"{self.config.ollama.host.rstrip('/')}/api/chat",
            {
                "model": self.config.model,
                "messages": messages,
                "stream": False,
                "keep_alive": self.config.ollama.keep_alive,
                "options": {
                    "temperature": self.config.temperature,
                    "num_ctx": self.config.ollama.num_ctx,
                },
            },
        )
        try:
            return str(body["message"]["content"])
        except (KeyError, TypeError) as exc:
            raise LlmError(f"Unexpected Ollama response: {str(body)[:300]}") from exc

    def _connection_hint(self) -> str:
        return (
            f"Cannot reach Ollama at {self.config.ollama.host}. "
            "Start the Ollama app or run: ollama serve"
        )

    def _error_message(self, response: httpx.Response) -> str:
        if response.status_code == 404:
            return (
                f"Model {self.config.model} is not available in Ollama. "
                f"Run: ollama pull {self.config.model}"
            )
        return super()._error_message(response)


class OpenAICompatibleClient(LlmClient):
    """Client for any API that implements OpenAI chat completions."""

    def __init__(
        self,
        config: LlmConfig,
        api_key: str | None,
        transport: httpx.BaseTransport | None = None,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        super().__init__(config, transport, retry_backoff_seconds)
        self._api_key = api_key

    def _send(self, messages: list[Message]) -> str:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else None
        body = self._post(
            f"{self.config.openai_compatible.base_url.rstrip('/')}/chat/completions",
            {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
            },
            headers,
        )
        try:
            return str(body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmError(f"Unexpected API response: {str(body)[:300]}") from exc

    def _connection_hint(self) -> str:
        return f"Cannot reach the LLM API at {self.config.openai_compatible.base_url}"

    def _error_message(self, response: httpx.Response) -> str:
        if response.status_code in (401, 403):
            return f"The LLM API rejected the credentials (HTTP {response.status_code}). " + (
                "Check LLM_API_KEY in .env."
            )
        return super()._error_message(response)


def create_llm_client(config: LlmConfig, secrets: Secrets) -> LlmClient:
    """Build the client for the configured backend."""
    if config.backend == "ollama":
        return OllamaClient(config)
    api_key = secrets.llm_api_key.get_secret_value() if secrets.llm_api_key else None
    return OpenAICompatibleClient(config, api_key)
