"""Lab 3 - a hybrid-retrieval RAG chatbot, with no embedding API required.

005_hybrid.ipynb builds a hybrid retriever - BM25 (lexical) + a VectorIndex
(semantic, via VoyageAI's embedding API) - fused with Reciprocal Rank Fusion, then
stops at retrieval. This package takes that same architecture and turns it into an
actual chatbot:

    retrieval.BM25Index    unchanged, exact keyword/ID matching
    retrieval.VectorIndex  unchanged, but embedding_fn is now embedder.TfidfEmbedder
                            (hand-rolled TF-IDF + cosine similarity) instead of
                            VoyageAI - no API key, no network call, just math/re
    retrieval.Retriever    unchanged, RRF fusion of both indexes
    bot.RagBot             new: retrieves fresh context every turn, then asks
                            Claude to answer grounded only in that context

The corpus (corpus.py) is three hardcoded synthetic reports - a fake vector DB,
not report.md.

Module layout:
    config.py      env loading, Anthropic client, cost/model constants
    retrieval.py   chunk_by_section, VectorIndex, BM25Index, Retriever
    embedder.py    TfidfEmbedder - the no-API-key semantic stand-in
    corpus.py      the hardcoded document corpus
    index.py       builds the retriever once at import time
    bot.py         RagBot - the retrieval-augmented chat loop
    cli.py         REPL + usage/search helpers
    main.py        this file - argv handling and entry point

Run it:

    python3 3-rag-and-agentic-search/rag-chatbot/main.py                        # REPL
    python3 3-rag-and-agentic-search/rag-chatbot/main.py "question"             # one-shot
    python3 3-rag-and-agentic-search/rag-chatbot/main.py --show-context "..."   # + retrieval trace

REPL commands: /reset  /usage  /quit  /debug  /search <query>
"""

import sys

from bot import RagBot
from cli import print_usage, repl

if __name__ == "__main__":
    bot = RagBot()

    args = sys.argv[1:]
    if args and args[0] in ("--show-context", "--debug"):
        bot.show_context = True
        args = args[1:]

    if args:  # one-shot mode: handy for scripted checks
        bot.chat(" ".join(args))
        print_usage(bot)
    else:
        repl(bot)
