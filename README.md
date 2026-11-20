# Oathbreaker AI for Hermes Agent

Use [Oathbreaker AI](https://oathbreaker.ai) as a model provider in [Hermes Agent](https://github.com/NousResearch/hermes-agent). Oathbreaker AI is a prepaid model API with an OpenAI-compatible Chat Completions endpoint. It serves abliterated GLM-5.3, DeepSeek V4.1 Flash and Kimi K3.1 Pro models, with streaming, function calling and reasoning.

## Install

```sh
hermes plugins install Oathbreaker-AI/hermes-oathbreaker --enable
```

The install asks for your API key and saves it to your Hermes `.env` as `OATHBREAKER_API_KEY`. Create a key in the [Oathbreaker console](https://oathbreaker.ai/dashboard); credit is prepaid, with no subscription.

Then pick the model:

```sh
hermes model        # choose "Oathbreaker AI", then a model
```

Or set it in `config.yaml`:

```yaml
model:
  provider: oathbreaker
  default: oathbreaker-glm-5.3
```

## Models

| Model | Name | Context window | Output per request |
|---|---|---|---|
| `oathbreaker-glm-5.3` (default) | GLM-5.3 Abliterated | 131,072 | up to 32,768 tokens |
| `oathbreaker-glm-5.3-flash` | GLM-5.3 Flash Abliterated | 131,072 | up to 32,768 tokens |
| `oathbreaker-deepseek-v4.1-flash` | DeepSeek V4.1 Flash Abliterated | 131,072 | up to 32,768 tokens |
| `oathbreaker-deepseek-v4.1-flash-fast` | DeepSeek V4.1 Flash Abliterated, faster serving | 131,072 | up to 32,768 tokens |
| `oathbreaker-kimi-k3.1-pro` | Kimi K3.1 Pro Abliterated | 131,072 | up to 24,576 tokens |

The context window is shared by the conversation and the requested output, and output limits include reasoning. Every model supports tools, streamed tool calls and reasoning; all are text-only. DeepSeek and Kimi sessions currently compact earlier than GLM sessions, because the API measures their window differently. Kimi K3.1 Pro can take 20 to 35 seconds to start answering; that is normal, not a hang. See the [API documentation](https://oathbreaker.ai/docs) for current limits and the [pricing page](https://oathbreaker.ai/pricing) for prices per million tokens.

`hermes model` lists the models your account can use; if the API cannot be reached, it offers these five. Compression and other auxiliary tasks use GLM-5.3 Flash; to change that, set the auxiliary model in your Hermes config.

## What the plugin sets

The API accepts only the request fields its [reference](https://oathbreaker.ai/docs#requests) lists, so the profile shapes Hermes's requests to match:

- **An output budget on every request:** 16,384 tokens unless you set `model.max_tokens` (up to 32,768, or 24,576 for Kimi K3.1 Pro). Without one, the API would limit replies to 8,192 tokens.
- **Reasoning effort:** `/reasoning` and `--reasoning` are sent as `reasoning_effort` with the API's three levels. `none`, `minimal`, `low` and `medium` send `low`; `high` sends `high`; `xhigh`, `max` and `ultra` send `max`. The mapping is the same for every model. The API has no reasoning-off setting, so `none` sends the lowest level. Left unset, the API's default for the model applies.
- **Conversation limits:** the API currently accepts up to 2,048 messages per request, with separate context and request-size limits that vary by model. When the API reports too many messages, Hermes compresses and retries. The handler also recognizes the older 256-message response.
- **Context limit:** when the API refuses a request as too long for the model's context (error code `context_limit`), Hermes compresses the conversation and retries instead of ending the turn. This works the same on every model.
- **Context:** every model declares its 131,072 window, so Hermes compresses before a request would be refused. Without this, Hermes would assume the DeepSeek and Kimi models have windows of about 1,000,000 tokens and never compress in time.

## What it sends where

The plugin registers one model provider and nothing else: no tools, hooks or commands. Your prompts, tool definitions and tool results go to `https://oathbreaker.ai/v1` through Hermes's own client, with your `OATHBREAKER_API_KEY`, which Hermes reads. The plugin itself opens no connections, reads and writes no files, and starts no processes. `OATHBREAKER_BASE_URL` overrides the endpoint.

## Limits to know

- Each account currently runs up to 10 generations at once and starts up to 60 requests a minute, shared across its keys. Check the [current rate limits](https://oathbreaker.ai/docs#rate-limits) before planning concurrent delegates or scheduled jobs.
- A generation can run for up to ten minutes.
- No image or audio input, and no embeddings.

## Tested

Version 1.1.0 passes `hermes plugins validate` and 93 regression cases on a Hermes Agent development build from November 2026 with the same provider interface. Version 1.0.1 was also checked on stock Hermes commit `77a30e9127998c0277306c361663f63c0694b88b` (November 15, 2026), where requests serialized by the real OpenAI SDK were checked against the API's own validator for streaming, tool results, reasoning levels, and explicit output budgets, and error recovery was checked using responses generated by that validator. These checks use synthetic requests and isolated profiles.

Regression tests in `tests/test_provider.py` load the plugin through Hermes's real provider discovery in a temporary profile. They cover the context window Hermes resolves for each model, the model list offered when the API cannot be reached, reasoning levels on every model, explicit output budgets, `context_limit` refusals (by error code, whatever the message says), current and legacy history limits, and errors that must not trigger compression. From a prepared Hermes source checkout, run:

```sh
HERMES_PYTHON=/path/to/hermes-test-environment/bin/python \
  scripts/run_tests.sh /path/to/hermes-oathbreaker/tests/test_provider.py
```

Version 1.1.0 adds the DeepSeek V4.1 Flash, DeepSeek V4.1 Flash fast and Kimi K3.1 Pro models, declares their 131,072 context windows, and recovers from `context_limit` refusals on every model. It retains the 16,384-token output default, GLM-5.3 Flash auxiliary model, and explicit user overrides. The separate Oathbreaker desktop app has its own 8,192-token and Flagship defaults.

## About

Operated by [Formacha LLC](https://oathbreaker.ai/about). Support: support@oathbreaker.ai. Not affiliated with Nous Research; Hermes Agent is their project.

MIT licensed.
