"""RagBot - a multi-turn conversation with Claude, grounded in fresh retrieval
each turn.
"""

from typing import Any, Dict, List, Tuple

from config import IN_PER_MTOK, MAX_TOKENS, MODEL, OUT_PER_MTOK, TOP_K, client
from index import retriever

SYSTEM = (
    "You are a retrieval-augmented assistant over a small demo knowledge base "
    "spanning cloud infrastructure, clinical research, and corporate finance "
    "documents. Answer the user's question using only the CONTEXT block provided "
    "with this turn - do not draw on outside knowledge, and do not guess or fill "
    "gaps. If the context doesn't contain the answer, say so plainly instead of "
    "speculating. When a fact comes from the context, briefly note which document "
    "(and section) it came from."
)


class RagBot:
    """chat() re-runs the hybrid retriever on every user message, builds a context
    block from the top results, and asks Claude to answer from that context alone.
    History lives in .messages and is resent every turn, complete with each turn's
    own already-baked-in context block.
    """

    def __init__(
        self,
        system: str = SYSTEM,
        model: str = MODEL,
        max_tokens: int = MAX_TOKENS,
        top_k: int = TOP_K,
    ):
        self.system = system
        self.model = model
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.show_context = False
        self.messages: List[dict] = []
        self.last_reply: str = ""
        self.input_tokens = 0
        self.output_tokens = 0

    def chat(self, user_message: str) -> None:
        results = retriever.search(user_message, k=self.top_k)

        if self.show_context:
            self._render_context_trace(results)

        context_block = self._build_context_block(results)
        composed = f"CONTEXT:\n{context_block}\n\nQUESTION: {user_message}"
        self.messages.append({"role": "user", "content": composed})

        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            messages=self.messages,
        )
        self._account(response)

        self.messages.append({"role": "assistant", "content": response.content})

        self.last_reply = "".join(b.text for b in response.content if b.type == "text")
        print(self.last_reply)

    def _build_context_block(self, results: List[Tuple[Dict[str, Any], float]]) -> str:
        if not results:
            return "(no matching context found)"
        entries = [
            f"[{i}] ({doc['source']} — {doc['section']})\n{doc['content']}"
            for i, (doc, _) in enumerate(results, start=1)
        ]
        return "\n\n".join(entries)

    def _render_context_trace(self, results: List[Tuple[Dict[str, Any], float]]) -> None:
        if not results:
            print("  [retrieved: nothing]")
            return
        print("  [retrieved]")
        for i, (doc, score) in enumerate(results, start=1):
            print(f"    {i}. score={score:.4f}  {doc['source']} — {doc['section']}")

    def _account(self, response) -> None:
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens

    def reset(self) -> None:
        """Forget the conversation. Keeps the system prompt and settings."""
        self.messages.clear()
        self.last_reply = ""

    def usage(self) -> dict:
        cost = (self.input_tokens * IN_PER_MTOK + self.output_tokens * OUT_PER_MTOK) / 1e6
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": cost,
        }
