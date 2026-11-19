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

    def complete(
        self, prompt: str, system: str | None = None, json_schema: dict[str, Any] | None = None
    ) -> str:
        """Return the model's reply to a single user prompt.

        With ``json_schema``, the backend is asked to reply with JSON only.
        """
        messages: list[Message] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        attempts = self.config.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                started = time.perf_counter()
                reply = self._send(messages, json_schema)
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
    def _send(self, messages: list[Message], json_schema: dict[str, Any] | None) -> str:
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

    def _send(self, messages: list[Message], json_schema: dict[str, Any] | None) -> str:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.config.ollama.keep_alive,
            "options": {
                "temperature": self.config.temperature,
                "num_ctx": self.config.ollama.num_ctx,
            },
        }
        if json_schema is not None:
            # Ollama constrains decoding to this schema.
            payload["format"] = json_schema
        body = self._post(f"{self.config.ollama.host.rstrip('/')}/api/chat", payload)
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
        key_source: str | None = None,
    ) -> None:
        super().__init__(config, transport, retry_backoff_seconds)
        self._api_key = api_key
        # Named in authentication errors so the user knows which key to fix.
        self._key_source = key_source or (
            "HF_TOKEN" if config.openai_compatible.use_hf_token else "LLM_API_KEY"
        )

    def _send(self, messages: list[Message], json_schema: dict[str, Any] | None) -> str:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else None
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        if json_schema is not None:
            # JSON mode is the most widely supported option; the prompt describes the shape.
            payload["response_format"] = {"type": "json_object"}
        body = self._post(
            f"{self.config.openai_compatible.base_url.rstrip('/')}/chat/completions",
            payload,
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
            return (
                f"The LLM API rejected the credentials (HTTP {response.status_code}): "
                f"{response.text[:200]}. Check {self._key_source} and its permissions."
            )
        return super()._error_message(response)


USER_KEY_SOURCE = "the API key entered in the demo"


def resolve_api_key(config: LlmConfig, secrets: Secrets) -> str | None:
    """Pick LLM_API_KEY, or HF_TOKEN when the config allows it."""
    if secrets.llm_api_key is not None:
        return secrets.llm_api_key.get_secret_value()
    if config.openai_compatible.use_hf_token and secrets.hf_token is not None:
        return secrets.hf_token.get_secret_value()
    return None


def same_endpoint(first: LlmConfig, second: LlmConfig) -> bool:
    """Whether two configs send requests to the same OpenAI-compatible endpoint."""
    if first.backend != "openai_compatible" or second.backend != "openai_compatible":
        return False

    def normalize(url: str) -> str:
        return url.strip().rstrip("/").lower()

    return normalize(first.openai_compatible.base_url) == normalize(
        second.openai_compatible.base_url
    )


def select_api_key(
    selected: LlmConfig,
    configured: LlmConfig,
    secrets: Secrets,
    user_api_key: str | None = None,
) -> str | None:
    """Pick the key for a request: the user's own key, else the server's key.

    The server's keys are used only when the request goes to the endpoint set in
    the config file, so a provider chosen in the demo can never receive them.
    """
    if user_api_key and user_api_key.strip():
        return user_api_key.strip()
    if same_endpoint(selected, configured):
        return resolve_api_key(configured, secrets)
    return None


def llm_unavailable_reason(config: LlmConfig, secrets: Secrets) -> str | None:
    """Explain why summaries cannot work with this configuration, or None if they can."""
    if config.backend == "openai_compatible" and resolve_api_key(config, secrets) is None:
        source = "HF_TOKEN" if config.openai_compatible.use_hf_token else "LLM_API_KEY"
        return f"Ringkasan tidak tersedia karena {source} belum diatur."
    return None


def create_llm_client(
    config: LlmConfig,
    secrets: Secrets,
    user_api_key: str | None = None,
    configured: LlmConfig | None = None,
) -> LlmClient:
    """Build the client for a backend.

    ``configured`` is the LLM section of the config file (defaults to ``config``);
    server keys are only sent when ``config`` points at the same endpoint.
    """
    if config.backend == "ollama":
        return OllamaClient(config)
    has_user_key = bool(user_api_key and user_api_key.strip())
    return OpenAICompatibleClient(
        config,
        select_api_key(config, configured or config, secrets, user_api_key),
        key_source=USER_KEY_SOURCE if has_user_key else None,
    )
