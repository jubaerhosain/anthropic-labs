"""Shared configuration: env loading, the Anthropic client, and cost constants."""

import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# .env lives at the repo root, three levels up from this file - load it by path so
# this script works no matter which directory you run it from.
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

if not os.getenv("ANTHROPIC_API_KEY"):
    raise RuntimeError(
        "ANTHROPIC_API_KEY not found. Copy .env.example to .env and paste your key in."
    )

# The client reads ANTHROPIC_API_KEY from the environment - don't pass it explicitly.
client = anthropic.Anthropic()

MODEL = "claude-haiku-4-5"  # cheapest model: $1 / $5 per million tokens
IN_PER_MTOK, OUT_PER_MTOK = 1.00, 5.00  # USD, claude-haiku-4-5
TOP_K = 4  # chunks pulled per turn
MAX_TOKENS = 1024  # grounded answers over a small corpus don't need more
