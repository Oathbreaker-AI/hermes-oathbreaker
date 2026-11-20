"""Oathbreaker wire and recovery contracts.

GLM rows were checked against the API on 2026-11-15; DeepSeek and Kimi rows follow the API's
published limits: a 131,072 context on every model, the same low/high/max effort vocabulary, and a
``context_limit`` refusal whose wording varies by model.
"""

import httpx
import openai
import pytest

from agent import models_dev
from agent.error_classifier import FailoverReason, classify_api_error
from agent.model_metadata import get_model_context_length
from agent.transports.chat_completions import ChatCompletionsTransport
from hermes_cli.models import provider_model_ids
from providers import get_provider_profile

MODELS = {
    "oathbreaker-glm-5.3": "glm",
    "oathbreaker-glm-5.3-flash": "glm",
    "oathbreaker-deepseek-v4.1-flash": "deepseek",
    "oathbreaker-deepseek-v4.1-flash-fast": "deepseek",
    "oathbreaker-kimi-k3.1-pro": "kimi",
}


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
    # No models.dev catalog: only the plugin's own declarations may decide a window.
    monkeypatch.setattr(models_dev, "_get_provider_models", lambda *_, **__: None, raising=False)
    token = set_hermes_home_override(home)
    try:
        result = get_provider_profile("oathbreaker")
        assert provider_source("oathbreaker") == "user"
        yield result
    finally:
        reset_hermes_home_override(token)


@pytest.mark.parametrize("model,family", MODELS.items())
def test_every_model_resolves_to_the_api_context_window(profile, model, family):
    metadata = models_dev.get_model_capabilities("oathbreaker", model, config={})
    assert metadata.context_window == 131_072
    assert metadata.supports_tools is True and metadata.supports_reasoning is True
    assert metadata.supports_vision is False and metadata.model_family == family
    # The window Hermes sizes compression by. Undeclared, the DeepSeek and Kimi ids fall to
    # name-based vendor defaults of about 1,000,000 tokens and the API refuses long sessions.
    assert get_model_context_length(model, base_url=profile.base_url, provider="oathbreaker") == 131_072


def test_offline_picker_offers_every_model_flagship_first(profile, monkeypatch):
    monkeypatch.setattr(profile, "fetch_models", lambda **_: None)  # GET /v1/models failed
    assert profile.fallback_models == tuple(MODELS)
    offered = provider_model_ids("oathbreaker")
    assert offered[0] == "oathbreaker-glm-5.3"
    assert set(MODELS) <= set(offered)
    for model in offered:  # nothing offered falls through to a name-based window
        assert get_model_context_length(model, base_url=profile.base_url, provider="oathbreaker") == 131_072
    assert profile.default_aux_model == "oathbreaker-glm-5.3-flash"


@pytest.mark.parametrize("model", MODELS)
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
    assert profile.supported_reasoning_efforts(model) == ("low", "high", "max")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("config,expected", [
    (None, None), ({"enabled": True}, None), ({"effort": "unknown"}, None),
    ({"enabled": False, "effort": "high"}, "low"),
])
def test_reasoning_defaults_and_explicit_output_budget_survive(profile, model, config, expected):
    # An unset effort stays off the wire, so each model gets the API's own default.
    kwargs = ChatCompletionsTransport().build_kwargs(
        model=model, messages=[{"role": "user", "content": "Short reply."}],
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


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("message", [
    "Input, requested output and safety headroom exceed the 131,072-token context limit. Compact the conversation or reduce max_tokens. No inference was sent.",
    "Messages and requested output exceed the conservative 131,072-unit admission budget (UTF-8 input bytes plus framing and requested output tokens).",
    "This request is too large for the model. Shorten it and try again.",  # a future wording
])
def test_context_limit_refusal_compresses_on_every_model(profile, model, message):
    verdict = classify_api_error(api_error(400, "context_limit", message), provider=profile.name,
                                 model=model, num_messages=40)
    assert verdict.reason == FailoverReason.context_overflow
    assert verdict.should_compress is True
    assert verdict.should_rotate_credential is False


@pytest.mark.parametrize("status,code,message", [
    (400, "invalid_request", "Provide between 1 and 2048 messages."),
    (400, "invalid_request", "The request contains an unsupported field."),
    (400, "invalid_tools", "Tool response does not match a pending call."),
    # Classification follows the code, not the wording: the same text under another code is not
    # an overflow, and neither is the code on a status other than 400.
    (400, "invalid_request", "Messages and requested output exceed the conservative 131,072-unit admission budget (UTF-8 input bytes plus framing and requested output tokens)."),
    (500, "context_limit", "This request is too large for the model. Shorten it and try again."),
    (401, "invalid_key", "Invalid API key."),
    (402, "insufficient_credit", "Insufficient credit."),
    (429, "rate_limit", "Too many requests. Try again later."),
])
def test_unrelated_failures_do_not_compress(profile, status, code, message):
    verdict = classify_api_error(api_error(status, code, message), provider=profile.name,
                                 model=profile.fallback_models[0])
    assert verdict.should_compress is False
