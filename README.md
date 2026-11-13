# Oathbreaker AI for Hermes Agent

Use [Oathbreaker AI](https://oathbreaker.ai) as a model provider in [Hermes Agent](https://github.com/NousResearch/hermes-agent). Oathbreaker AI is a prepaid model API with an OpenAI-compatible Chat Completions endpoint. It serves GLM-5.3 Abliterated and GLM-5.3 Flash Abliterated, with streaming and function calling.

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
| `oathbreaker-glm-5.3` | GLM-5.3 Abliterated | 131,072 tokens | up to 32,768 tokens, reasoning included |
| `oathbreaker-glm-5.3-flash` | GLM-5.3 Flash Abliterated | 131,072 tokens | up to 32,768 tokens, reasoning included |

Both models are text-only. Prices per million tokens are on the [pricing page](https://oathbreaker.ai/pricing). Compression and other auxiliary tasks use Flash; to change that, set the auxiliary model in your Hermes config.

## What the plugin sets

The API accepts only the request fields its [reference](https://oathbreaker.ai/docs#requests) lists, so the profile shapes Hermes's requests to match:

- **An output budget on every request:** 16,384 tokens unless you set `model.max_tokens` (up to 32,768). Without one, Flash answers with at most 256 tokens.
- **Reasoning effort:** `/reasoning` and `--reasoning` are sent as `reasoning_effort` with the API's three levels. `none`, `minimal`, `low` and `medium` send `low`; `high` sends `high`; `xhigh`, `max` and `ultra` send `max`. GLM-5.3 always reasons, so `none` means the lowest level. Left unset, the API's default applies.
- **The 256-message limit:** the API takes at most 256 messages per request. When a long session reaches it, Hermes compresses the conversation and retries instead of stopping.
- **Context:** both models declare a 131,072-token window, so Hermes compresses before a request would be refused.

## What it sends where

The plugin registers one model provider and nothing else: no tools, hooks or commands. Your prompts, tool definitions and tool results go to `https://oathbreaker.ai/v1` through Hermes's own client, with your `OATHBREAKER_API_KEY`, which Hermes reads. The plugin itself opens no connections, reads and writes no files, and starts no processes. `OATHBREAKER_BASE_URL` overrides the endpoint.

## Limits to know

- Each account runs 2 generations at once and starts up to 20 requests a minute, shared across its keys. Running many delegates or scheduled jobs in parallel will hit this.
- A generation can run for up to ten minutes.
- No image or audio input, and no embeddings.

## Tested

Checked with `hermes plugins validate`, and against a local stand-in server that records each request: plain chat, a tool call and its follow-up, both models, and each reasoning level. Every recorded request passes the API's own request validator. Hermes's error classifier was also run on the API's real error bodies: the 256-message and context-limit errors both come back as "compress and retry". Tested on Hermes 0.23.1's provider, transport and error-classification code.

## About

Operated by [Formacha LLC](https://oathbreaker.ai/about). Support: support@oathbreaker.ai. Not affiliated with Nous Research; Hermes Agent is their project.

MIT licensed.
