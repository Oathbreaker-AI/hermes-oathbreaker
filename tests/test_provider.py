"""Oathbreaker wire and recovery contracts, checked against the API on 2026-11-15."""

import httpx
import openai
import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from agent.transports.chat_completions import ChatCompletionsTransport
from providers import get_provider_profile


@pytest.fixture
def profile(tmp_path, monkeypatch):
    import shutil
    from pathlib import Path

    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from providers import provider_source

    home = tmp_path / "hermes"
    plugins = home / "plugins"
    plugins.mkdir(parents=True)
    shutil.copytree(Path(__file__).resolve().parents[1], plugins / "oathbreaker-provider",
                    ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "tests"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    token = set_hermes_home_override(home)
    try:
        result = get_provider_profile("oathbreaker")
        assert provider_source("oathbreaker") == "user"
        yield result
    finally:
        reset_hermes_home_override(token)


@pytest.mark.parametrize("model", ["oathbreaker-glm-5.3", "oathbreaker-glm-5.3-flash"])
@pytest.mark.parametrize("effort,expected", [
    ("none", "low"), ("minimal", "low"), ("low", "low"), ("medium", "low"),
    ("high", "high"), ("xhigh", "max"), ("max", "max"), ("ultra", "max"),
])
def test_reasoning_uses_the_api_wire_vocabulary(profile, model, effort, expected):
    kwargs = ChatCompletionsTransport().build_kwargs(
        model=model, messages=[{"role": "user", "content": "Use the requested effort."}],
        provider_profile=profile, base_url=profile.base_url, supports_reasoning=True,
        reasoning_config={"enabled": True, "effort": effort},
        max_tokens_param_fn=lambda n: {"max_tokens": n},
    )
    assert kwargs["reasoning_effort"] == expected
    assert "reasoning" not in kwargs and "reasoning" not in kwargs.get("extra_body", {})
    assert kwargs["max_tokens"] == profile.get_max_tokens(model)
    assert profile.setup_reasoning_efforts(model) == ("low", "high", "max")


@pytest.mark.parametrize("config,expected", [
    (None, None), ({"enabled": True}, None), ({"effort": "unknown"}, None),
    ({"enabled": False, "effort": "high"}, "low"),
])
def test_reasoning_defaults_and_explicit_output_budget_survive(profile, config, expected):
    kwargs = ChatCompletionsTransport().build_kwargs(
        model=profile.fallback_models[0], messages=[{"role": "user", "content": "Short reply."}],
        provider_profile=profile, base_url=profile.base_url, reasoning_config=config,
        max_tokens=512, max_tokens_param_fn=lambda n: {"max_tokens": n},
    )
    assert kwargs.get("reasoning_effort") == expected
    assert kwargs["max_tokens"] == 512


def api_error(status, code, message):
    body = {"error": {"code": code, "message": message}}
    response = httpx.Response(status, json=body, request=httpx.Request(
        "POST", "https://oathbreaker.ai/v1/chat/completions"))
    return openai.APIStatusError(message, response=response, body=body)


@pytest.mark.parametrize("code,message", [
    ("invalid_request", "Conversation has 2049 messages; the maximum is 2048. Compact the conversation or start a new chat. No inference was sent or credits reserved."),
    ("invalid_request", "Conversation has 513 messages; the maximum is 512. Compact the conversation or start a new chat. No inference was sent or credits reserved."),
    ("invalid_request", "Provide between 1 and 256 messages."),
    ("context_limit", "Input, requested output and safety headroom exceed the 131,072-token context limit. Compact the conversation or reduce max_tokens. No inference was sent."),
])
def test_oversized_history_uses_compression_recovery(profile, code, message):
    verdict = classify_api_error(api_error(400, code, message), provider=profile.name,
                                 model=profile.fallback_models[0], num_messages=2049)
    assert verdict.reason == FailoverReason.context_overflow
    assert verdict.should_compress is True
    assert verdict.should_rotate_credential is False


@pytest.mark.parametrize("status,code,message", [
    (400, "invalid_request", "Provide between 1 and 2048 messages."),
    (400, "invalid_request", "The request contains an unsupported field."),
    (400, "invalid_tools", "Tool response does not match a pending call."),
    (401, "invalid_key", "Invalid API key."),
    (402, "insufficient_credit", "Insufficient credit."),
    (429, "rate_limit", "Too many requests. Try again later."),
])
def test_unrelated_failures_do_not_compress(profile, status, code, message):
    verdict = classify_api_error(api_error(status, code, message), provider=profile.name,
                                 model=profile.fallback_models[0])
    assert verdict.should_compress is False
