"""REPL and one-shot CLI plumbing for RagBot."""

import anthropic

from bot import RagBot
from config import MODEL, TOP_K
from index import retriever


def print_usage(bot: RagBot) -> None:
    u = bot.usage()
    print(f"  {u['input_tokens']:>7} in  {u['output_tokens']:>6} out  ~${u['cost_usd']:.5f}")


def print_search(query: str, k: int = TOP_K) -> None:
    results = retriever.search(query, k=k)
    if not results:
        print("  [no results]")
        return
    for i, (doc, score) in enumerate(results, start=1):
        preview = doc["content"].replace("\n", " ")[:100]
        print(f"  {i}. score={score:.4f}  {doc['source']} — {doc['section']}")
        print(f"     {preview}...")


def repl(bot: RagBot) -> None:
    print(f"Lab 3 - hybrid RAG chatbot on {MODEL}. Commands: /reset  /usage  /debug  /search <q>  /quit\n")

    while True:
        try:
            user_message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_message:
            continue
        if user_message == "/quit":
            break
        if user_message == "/reset":
            bot.reset()
            print("  [history cleared]")
            continue
        if user_message == "/usage":
            print_usage(bot)
            continue
        if user_message == "/debug":
            bot.show_context = not bot.show_context
            print(f"  [context trace {'on' if bot.show_context else 'off'}]")
            continue
        if user_message.startswith("/search "):
            print_search(user_message[len("/search ") :].strip())
            continue

        try:
            bot.chat(user_message)
        except anthropic.APIError as exc:
            print(f"  [api error: {exc}]")

    print_usage(bot)
