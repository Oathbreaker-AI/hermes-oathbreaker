"""Oathbreaker AI as a Hermes model provider.

Oathbreaker AI (https://oathbreaker.ai) serves abliterated GLM-5.3, DeepSeek V4.1 Flash and
Kimi K3.1 Pro models through a prepaid, OpenAI-compatible Chat Completions API. This profile
registers it with Hermes's own client: requests go to https://oathbreaker.ai/v1 with
OATHBREAKER_API_KEY, which Hermes reads. The plugin opens no connections itself, reads and writes
no files, and starts no processes.

What the profile adjusts, and why (the API accepts only the request fields it documents):
  - an output budget on every request, so replies are not held to the API's 8,192-token default;
  - a reasoning choice is sent as top-level ``reasoning_effort`` (low, high or max) instead of
    Hermes's generic ``reasoning`` field, which the API rejects;
  - a ``context_limit`` refusal, and a message-count overflow, count as context overflow, so
    Hermes compresses the conversation and retries instead of failing, including responses from
    older API deployments;
  - every model declares the API's 131,072 context window. Without that, Hermes would give the
    DeepSeek and Kimi ids their vendors' much larger windows and never compress in time.
"""

import re
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

FLAGSHIP = "oathbreaker-glm-5.3"
FLASH = "oathbreaker-glm-5.3-flash"
DEEPSEEK_FLASH = "oathbreaker-deepseek-v4.1-flash"
DEEPSEEK_FLASH_FAST = "oathbreaker-deepseek-v4.1-flash-fast"
KIMI = "oathbreaker-kimi-k3.1-pro"
# Model id -> family, in picker order: the flagship leads and stays the default.
MODEL_FAMILIES = {
    FLAGSHIP: "glm",
    FLASH: "glm",
    DEEPSEEK_FLASH: "deepseek",
    DEEPSEEK_FLASH_FAST: "deepseek",
    KIMI: "kimi",
}
CONTEXT_WINDOW = 131_072
EFFORTS = ("low", "high", "max")
_HISTORY_LIMIT = re.compile(r"\bconversation has ([1-9][0-9]*) messages; the maximum is ([1-9][0-9]*)\.")

# Hermes's effort ladder onto the API's three levels, the same on every model. Never escalates,
# except xhigh -> max, as Hermes does for GLM-5.3 elsewhere.
_EFFORT = {
    "none": "low", "minimal": "low", "low": "low", "medium": "low",
    "high": "high", "xhigh": "max", "max": "max", "ultra": "max",
}


def _reasoning_effort(reasoning_config: dict | None) -> str | None:
    """The API's ``reasoning_effort`` for Hermes's reasoning config, or None to leave it unset."""
    if not isinstance(reasoning_config, dict):
        return None
    if reasoning_config.get("enabled") is False:
        return "low"  # the API has no reasoning-off value; low is the floor on every model
    effort = str(reasoning_config.get("effort") or "").strip().lower()
    return _EFFORT.get(effort) if effort else None


def _classify_api_error(error, *, status_code=None, error_code=None, message="", body=None, model=None):
    """Recover a context or message-count overflow without treating malformed input as a full history."""
    if status_code != 400:
        return None
    if error_code == "context_limit":
        # The API's refusal of a request too large for the model's context, on every model. Decided
        # by the code alone: the message wording differs between models and may change.
        return {"reason": "context_overflow", "should_compress": True}
    if error_code not in (None, "invalid_request"):
        return None
    text = (message or "").lower()
    count = _HISTORY_LIMIT.search(text)
    if (count and int(count[1]) > int(count[2])) or "provide between 1 and 256 messages." in text:
        return {"reason": "context_overflow", "should_compress": True}
    return None


class OathbreakerProfile(ProviderProfile):
    """oathbreaker.ai: abliterated GLM-5.3, DeepSeek V4.1 Flash and Kimi K3.1 Pro, OpenAI-compatible."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        effort = _reasoning_effort(reasoning_config)
        return {}, ({"reasoning_effort": effort} if effort else {})

    def supported_reasoning_efforts(self, model: str | None) -> tuple[str, ...]:
        return EFFORTS


oathbreaker = OathbreakerProfile(
    name="oathbreaker",
    aliases=("oathbreaker-ai", "oathbreaker.ai"),
    display_name="Oathbreaker AI",
    description=(
        "Oathbreaker AI: abliterated GLM-5.3, DeepSeek V4.1 Flash and Kimi K3.1 Pro "
        "through a prepaid OpenAI-compatible API"
    ),
    signup_url="https://oathbreaker.ai/agents",
    env_vars=("OATHBREAKER_API_KEY", "OATHBREAKER_BASE_URL"),
    base_url="https://oathbreaker.ai/v1",
    hostnames=("oathbreaker.ai",),
    auth_type="api_key",
    default_aux_model=FLASH,
    fallback_models=tuple(MODEL_FAMILIES),
    default_max_tokens=16_384,
    classify_api_error=_classify_api_error,
    setup_reasoning_efforts=lambda model: EFFORTS,
    model_capabilities={
        model: {
            "context_window": CONTEXT_WINDOW,
            "supports_tools": True,
            "supports_vision": False,
            "supports_reasoning": True,
            "model_family": family,
        }
        for model, family in MODEL_FAMILIES.items()
    },
)

register_provider(oathbreaker)
