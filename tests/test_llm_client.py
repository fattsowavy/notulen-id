"""Tests for the LLM clients, using a mocked HTTP transport."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest
from pydantic import SecretStr

from notulen.config import LlmConfig, OllamaConfig, OpenAICompatibleConfig, Secrets
from notulen.summarize.llm_client import (
    LlmError,
    OllamaClient,
    OpenAICompatibleClient,
    create_llm_client,
)

CONFIG = LlmConfig(
    backend="ollama",
    model="gemma3:4b",
    temperature=0.2,
    timeout_seconds=30,
    max_retries=2,
    ollama=OllamaConfig(host="http://ollama.test:11434/", num_ctx=8192),
    openai_compatible=OpenAICompatibleConfig(base_url="https://api.test/v1/"),
)

Handler = Callable[[httpx.Request], httpx.Response]


def recording(responses: list[httpx.Response], requests: list[httpx.Request]) -> Handler:
    """Return a handler that records requests and replays responses in order."""

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return responses[min(len(requests), len(responses)) - 1]

    return handle


def ollama_client(handler: Handler) -> OllamaClient:
    return OllamaClient(CONFIG, transport=httpx.MockTransport(handler), retry_backoff_seconds=0)


def ollama_reply(text: str) -> httpx.Response:
    return httpx.Response(200, json={"message": {"role": "assistant", "content": text}})


def test_ollama_request_format() -> None:
    requests: list[httpx.Request] = []
    client = ollama_client(recording([ollama_reply("Ringkasan")], requests))

    assert client.complete("Ringkas ini", system="Kamu notulis") == "Ringkasan"

    request = requests[0]
    body = json.loads(request.content)
    assert str(request.url) == "http://ollama.test:11434/api/chat"
    assert body["model"] == "gemma3:4b"
    assert body["stream"] is False
    assert body["keep_alive"] == "5m"
    assert body["options"] == {"temperature": 0.2, "num_ctx": 8192}
    assert [message["role"] for message in body["messages"]] == ["system", "user"]


def test_json_schema_is_sent_to_each_backend() -> None:
    schema = {"type": "object", "properties": {"topik": {"type": "array"}}}
    ollama_requests: list[httpx.Request] = []
    openai_requests: list[httpx.Request] = []
    openai_reply = httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    ollama_client(recording([ollama_reply("{}")], ollama_requests)).complete(
        "halo", json_schema=schema
    )
    OpenAICompatibleClient(
        CONFIG, None, transport=httpx.MockTransport(recording([openai_reply], openai_requests))
    ).complete("halo", json_schema=schema)

    assert json.loads(ollama_requests[0].content)["format"] == schema
    assert json.loads(openai_requests[0].content)["response_format"] == {"type": "json_object"}


def test_plain_requests_do_not_ask_for_json() -> None:
    requests: list[httpx.Request] = []

    ollama_client(recording([ollama_reply("teks")], requests)).complete("halo")

    assert "format" not in json.loads(requests[0].content)


def test_server_errors_are_retried() -> None:
    requests: list[httpx.Request] = []
    responses = [httpx.Response(500, text="busy"), ollama_reply("berhasil")]

    assert ollama_client(recording(responses, requests)).complete("halo") == "berhasil"
    assert len(requests) == 2


def test_retries_stop_after_the_configured_limit() -> None:
    requests: list[httpx.Request] = []
    client = ollama_client(recording([httpx.Response(503, text="down")], requests))

    with pytest.raises(LlmError, match="3 attempt"):
        client.complete("halo")
    assert len(requests) == 3


def test_missing_ollama_model_explains_how_to_pull() -> None:
    requests: list[httpx.Request] = []
    response = httpx.Response(404, json={"error": "model 'gemma3:4b' not found"})

    with pytest.raises(LlmError, match="ollama pull gemma3:4b"):
        ollama_client(recording([response], requests)).complete("halo")
    assert len(requests) == 1


def test_unreachable_ollama_explains_how_to_start_it() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(LlmError, match="ollama serve"):
        ollama_client(refuse).complete("halo")


@pytest.mark.parametrize(
    "response",
    [
        ollama_reply("   "),
        httpx.Response(200, json={"unexpected": True}),
        httpx.Response(200, text="not json"),
    ],
)
def test_unusable_responses_raise(response: httpx.Response) -> None:
    with pytest.raises(LlmError):
        ollama_client(recording([response], [])).complete("halo")


def test_openai_compatible_request_format() -> None:
    requests: list[httpx.Request] = []
    reply = httpx.Response(200, json={"choices": [{"message": {"content": "Ringkasan"}}]})
    client = OpenAICompatibleClient(
        CONFIG, "sk-test", transport=httpx.MockTransport(recording([reply], requests))
    )

    assert client.complete("Ringkas ini") == "Ringkasan"

    request = requests[0]
    assert str(request.url) == "https://api.test/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer sk-test"
    assert json.loads(request.content)["model"] == "gemma3:4b"


def test_openai_compatible_without_key_sends_no_auth_header() -> None:
    requests: list[httpx.Request] = []
    reply = httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
    client = OpenAICompatibleClient(
        CONFIG, None, transport=httpx.MockTransport(recording([reply], requests))
    )

    client.complete("halo")

    assert "Authorization" not in requests[0].headers


def test_rejected_credentials_mention_the_env_variable() -> None:
    client = OpenAICompatibleClient(
        CONFIG,
        "sk-wrong",
        transport=httpx.MockTransport(recording([httpx.Response(401, text="no")], [])),
    )

    with pytest.raises(LlmError, match="LLM_API_KEY"):
        client.complete("halo")


def test_factory_selects_backend() -> None:
    secrets = Secrets(llm_api_key=SecretStr("sk-test"))
    openai_config = CONFIG.model_copy(update={"backend": "openai_compatible"})

    assert isinstance(create_llm_client(CONFIG, secrets), OllamaClient)
    assert isinstance(create_llm_client(openai_config, secrets), OpenAICompatibleClient)
