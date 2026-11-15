"""Oathbreaker AI as a Hermes model provider.

Oathbreaker AI (https://oathbreaker.ai) serves GLM-5.3 Abliterated and GLM-5.3 Flash Abliterated
through a prepaid, OpenAI-compatible Chat Completions API. This profile registers it with
Hermes's own client: requests go to https://oathbreaker.ai/v1 with OATHBREAKER_API_KEY, which
Hermes reads. The plugin opens no connections itself, reads and writes no files, and starts no
processes.

What the profile adjusts, and why (the API accepts only the request fields it documents):
  - an output budget on every request: Flash defaults to 256 output tokens when none is sent;
  - a reasoning choice is sent as top-level ``reasoning_effort`` (low, high or max) instead of
    Hermes's generic ``reasoning`` field, which the API rejects;
  - message-count overflow counts as context overflow, so Hermes compresses the
    conversation instead of failing, including responses from older API deployments;
  - both models declare their 131,072-token context window.
"""

import re
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

FLAGSHIP = "oathbreaker-glm-5.3"
FLASH = "oathbreaker-glm-5.3-flash"
CONTEXT_WINDOW = 131_072
EFFORTS = ("low", "high", "max")
_HISTORY_LIMIT = re.compile(r"\bconversation has ([1-9][0-9]*) messages; the maximum is ([1-9][0-9]*)\.")

# Hermes's effort ladder onto the API's three levels. Never escalates, except xhigh -> max, as
# Hermes does for GLM-5.3 elsewhere.
_EFFORT = {
    "none": "low", "minimal": "low", "low": "low", "medium": "low",
    "high": "high", "xhigh": "max", "max": "max", "ultra": "max",
}


def _reasoning_effort(reasoning_config: dict | None) -> str | None:
    """The API's ``reasoning_effort`` for Hermes's reasoning config, or None to leave it unset."""
    if not isinstance(reasoning_config, dict):
        return None
    if reasoning_config.get("enabled") is False:
        return "low"  # GLM-5.3 always reasons; low is the floor
    effort = str(reasoning_config.get("effort") or "").strip().lower()
    return _EFFORT.get(effort) if effort else None


def _classify_api_error(error, *, status_code=None, error_code=None, message="", body=None, model=None):
    """Recover a message-count overflow without treating malformed input as a full history."""
    if status_code != 400 or error_code not in (None, "invalid_request"):
        return None
    text = (message or "").lower()
    count = _HISTORY_LIMIT.search(text)
    if (count and int(count[1]) > int(count[2])) or "provide between 1 and 256 messages." in text:
        return {"reason": "context_overflow", "should_compress": True}
    return None


class OathbreakerProfile(ProviderProfile):
    """oathbreaker.ai: GLM-5.3 Abliterated, OpenAI-compatible Chat Completions."""

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
    description="Oathbreaker AI: GLM-5.3 Abliterated through a prepaid OpenAI-compatible API",
    signup_url="https://oathbreaker.ai/agents",
    env_vars=("OATHBREAKER_API_KEY", "OATHBREAKER_BASE_URL"),
    base_url="https://oathbreaker.ai/v1",
    hostnames=("oathbreaker.ai",),
    auth_type="api_key",
    default_aux_model=FLASH,
    fallback_models=(FLAGSHIP, FLASH),
    default_max_tokens=16_384,
    classify_api_error=_classify_api_error,
    setup_reasoning_efforts=lambda model: EFFORTS,
    model_capabilities={
        model: {
            "context_window": CONTEXT_WINDOW,
            "supports_tools": True,
            "supports_vision": False,
            "supports_reasoning": True,
            "model_family": "glm",
        }
        for model in (FLAGSHIP, FLASH)
    },
)

register_provider(oathbreaker)
