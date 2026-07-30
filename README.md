# anthropic-labs

Hands-on notebooks for learning the Claude API, one concept at a time.

## What this is

A lab notebook series built directly on the Messages API with the official `anthropic`
Python SDK — no framework, no wrapper. Each lab is a standalone Jupyter notebook that runs
on the cheapest model that can demonstrate the concept, so working through one costs a
fraction of a cent.

## Labs

| # | Notebook | What it covers |
|---|---|---|
| 1 | [`1-chatbot.ipynb`](1-chatbot.ipynb) | Messages API basics, streaming, multi-turn history, system prompts, token and cost accounting |

Planned, not yet written: tool use, prompt caching, structured outputs, the Batch API.

## Setup

```bash
pip install -r requirements.txt   # anthropic, python-dotenv, ipykernel
cp .env.example .env              # then paste your key into .env
jupyter lab                       # or open the notebook in VS Code / Cursor
```

Get a key at [platform.claude.com/settings/keys](https://platform.claude.com/settings/keys).

## A note on your API key

`.env` is gitignored, so the key stays on your machine. Never paste a key into a notebook
cell — cell contents get committed. The SDK reads `ANTHROPIC_API_KEY` from the environment,
so construct the client with no arguments rather than passing the key explicitly.

## Cost

Lab 1 runs on **Claude Haiku 4.5** (`claude-haiku-4-5`) — $1 per million input tokens, $5
per million output. A long session costs a fraction of a cent.

The Messages API is stateless: the whole conversation history is resent on every turn, so
input tokens grow with each turn. That's the main cost driver in a long conversation, and
the reason prompt caching exists.

Swapping `MODEL` in the notebook buys more capability at higher cost:

| Model | Model ID | Context | Input $/MTok | Output $/MTok |
|---|---|---|---|---|
| Claude Haiku 4.5 | `claude-haiku-4-5` | 200K | $1.00 | $5.00 |
| Claude Sonnet 5 | `claude-sonnet-5` | 1M | $3.00 | $15.00 |
| Claude Opus 5 | `claude-opus-5` | 1M | $5.00 | $25.00 |

Sonnet 5 and Opus 5 also support the `effort` parameter, which Haiku 4.5 does not — sending
it to Haiku returns a 400. These numbers move; check the
[pricing page](https://platform.claude.com/docs/en/pricing) for current rates.

## Requirements

Python 3.10+ (the notebooks use `str | None` union syntax) and an Anthropic API key.
