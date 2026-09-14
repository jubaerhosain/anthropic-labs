"""Multi-tool agent loop - driven by stop_reason, not by an iteration counter.

Two client-side tools are wired into one loop:

    calculator    we define it, we execute it
    web_search    we define it, we execute it   (a STUB - canned results, no network)

Both are client tools. That's the difference from lab 2, which used Anthropic's
server-side `web_search`: a server tool runs mid-turn on Anthropic's infrastructure and
its results are already in the response. A client tool stops the turn - the API returns
with `stop_reason: "tool_use"`, and nothing continues until we run the function and send
a `tool_result` back.

The loop's control flow is `stop_reason` and nothing else:

    end_turn    -> Claude is done. Extract the text and return it.
    tool_use    -> Claude wants a tool. Run it, append the result, go around again.
    other       -> max_tokens / refusal / stop_sequence: terminal, handle and return.

MAX_ITERATIONS is a safety cap, not the stopping mechanism. `for _ in range(cap)` makes
the cap the control flow - the loop ends because it ran out of turns, which is an
accident, not a decision. `while True` on stop_reason ends because Claude said it was
done. The cap below should never fire; if it does, that's a bug worth a warning.

Run it:

    python3 ccar-f/agentic-loops/agent_loop.py                  # sequential demo
    python3 ccar-f/agentic-loops/agent_loop.py "what is 7**12"  # your own prompt
"""

import ast
import json
import logging
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# .env lives at the repo root, three levels up - load it by path so this script works
# no matter which directory you run it from.
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

if not os.getenv("ANTHROPIC_API_KEY"):
    raise RuntimeError(
        "ANTHROPIC_API_KEY not found. Copy .env.example to .env and paste your key in."
    )

# The client reads ANTHROPIC_API_KEY from the environment - don't pass it explicitly.
client = anthropic.Anthropic()

MODEL = "claude-haiku-4-5"  # cheapest model: $1 / $5 per million tokens
MAX_TOKENS = 2048

# A backstop, NOT the stopping mechanism. Every normal query terminates on
# stop_reason == "end_turn" after 2-4 iterations, far below this.
MAX_ITERATIONS = 20

# WARNING, not INFO - the SDK's httpx logger would otherwise print a line per request.
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)
log = logging.getLogger("agent_loop")


# ---------------------------------------------------------------------------
# 1. Tool definitions
# ---------------------------------------------------------------------------
# Raw JSON Schema, no framework. The description is what makes Claude reach for a
# tool, so say *when* to call it, not just what it does.

CALCULATOR = {
    "name": "calculator",
    "description": (
        "Evaluate an arithmetic expression and return the exact result. Call this "
        "whenever a question needs a precise number rather than an estimate - mental "
        "arithmetic is unreliable, especially for large numbers and division."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": (
                    "An arithmetic expression using + - * / // % ** and parentheses, "
                    "e.g. '(1247 * 89) / 3'. Numbers and operators only - no variables, "
                    "no function calls."
                ),
            }
        },
        "required": ["expression"],
    },
}

# A *client* tool named web_search - note the plain name/description/input_schema and
# the absence of a `type` field. Anthropic's server-side web search is declared as
# {"type": "web_search_20250305", "name": "web_search"} and never reaches this process;
# this one is ours to execute.
WEB_SEARCH = {
    "name": "web_search",
    "description": (
        "Search the web for facts you don't already know - populations, prices, dates, "
        "measurements. Returns short text snippets. Search first, then compute with the "
        "numbers you find rather than guessing at them."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query, e.g. 'Dhaka metro population 2024'.",
            },
            "max_results": {
                "type": "integer",
                "description": "How many snippets to return. Defaults to 3.",
            },
        },
        "required": ["query"],
    },
}

TOOLS = [CALCULATOR, WEB_SEARCH]


# ---------------------------------------------------------------------------
# 2. The client-side implementations
# ---------------------------------------------------------------------------

# Whitelist for the calculator. Tool input is model output, which is untrusted input -
# bare eval() on it would hand the model arbitrary code execution.
_ALLOWED_NODES = (
    ast.Expression,
    ast.Constant,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
)


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, rejecting anything that isn't arithmetic."""
    tree = ast.parse(expression, mode="eval")

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(
                f"{type(node).__name__} is not allowed - arithmetic on numbers only."
            )
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError("only numeric literals are allowed.")
        # 9**9**9 is valid arithmetic that hangs the process - a whitelist alone
        # isn't enough, the values matter too.
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            if not isinstance(node.right, ast.Constant) or abs(node.right.value) > 1000:
                raise ValueError("exponent must be a literal no larger than 1000.")

    return str(eval(compile(tree, "<calculator>", "eval")))  # noqa: S307 - tree is validated


# Canned corpus. A real implementation would call a search API here; the shape of the
# loop is identical either way. Every snippet carries a concrete number, so Claude has
# something to hand the calculator on the next iteration.
_CORPUS = {
    ("dhaka", "population"): [
        "Dhaka's metro population reached 23,900,000 in 2024 (UN DESA World Urbanization "
        "Prospects).",
        "Dhaka added roughly 900,000 residents between 2023 and 2024, the fastest "
        "absolute growth of any South Asian metro.",
    ],
    ("bangladesh", "growth"): [
        "Bangladesh's annual population growth rate was 1.03% in 2024, down from 1.12% "
        "in 2023.",
        "The country's total population stood at 173,600,000 in mid-2024.",
    ],
    ("tokyo", "population"): [
        "Tokyo's metro population was 37,100,000 in 2024, a decline of about 0.2% "
        "year over year.",
    ],
    ("gold", "price"): [
        "Gold closed at $2,063.40 per troy ounce on 2 January 2024.",
        "One troy ounce is 31.1035 grams.",
    ],
    ("earth", "moon"): [
        "The mean distance from Earth to the Moon is 384,400 km.",
        "Light covers 299,792 km per second in vacuum.",
    ],
    ("everest", "height"): [
        "Mount Everest's summit is 8,848.86 m above sea level (2020 China-Nepal survey).",
    ],
}

_STUB_PREFIX = "[stub results - canned data, not a live search]"


def web_search(query: str, max_results: int = 3) -> str:
    """Return canned snippets for a query. No network call is made."""
    words = set(query.lower().replace("'s", "").split())

    snippets: list[str] = []
    for keywords, entries in _CORPUS.items():
        if all(any(k in w for w in words) for k in keywords):
            snippets.extend(entries)

    if not snippets:
        return f"{_STUB_PREFIX}\nNo results for {query!r}. Try different search terms."

    numbered = "\n".join(
        f"{i}. {s}" for i, s in enumerate(snippets[: max(1, max_results)], start=1)
    )
    return f"{_STUB_PREFIX}\n{numbered}"


def dispatch(name: str, tool_input: dict) -> tuple[str, bool]:
    """Run a client-side tool. Returns (content, is_error).

    Failures come back as an error *result*, never an exception. Claude sees the
    message and can retry with different input; raising here would kill the loop.
    """
    try:
        if name == "calculator":
            return calculator(**tool_input), False
        if name == "web_search":
            return web_search(**tool_input), False
        return f"No tool named {name!r}.", True
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}", True


# ---------------------------------------------------------------------------
# 3. The agentic loop
# ---------------------------------------------------------------------------

SYSTEM = (
    "You are a research assistant with two tools: web_search for facts you don't know, "
    "and calculator for exact arithmetic. When a question needs a looked-up number and "
    "then a computation, search first and feed the real figure into the calculator - "
    "never estimate a number you could look up, and never do the arithmetic in your "
    "head. Search results are canned demo data, so say so when you report them. "
    "Answer briefly."
)


def text_of(response) -> str:
    """Concatenate the text blocks of a response, ignoring tool_use blocks."""
    return "".join(b.text for b in response.content if b.type == "text").strip()


def render(response, iteration: int) -> None:
    """Print one line per content block so the loop is observable."""
    for block in response.content:
        if block.type == "text":
            if block.text.strip():
                print(f"  [iter {iteration}] {block.text.strip()}")
        elif block.type == "tool_use":
            print(f"  [iter {iteration}] tool: {block.name}({json.dumps(block.input)})")


def handle_terminal_stop(response, iteration: int) -> str:
    """Stop reasons other than end_turn / tool_use. All of them end the loop."""
    reason = response.stop_reason

    if reason == "max_tokens":
        print(f"  [iter {iteration}] truncated at max_tokens - raise MAX_TOKENS]")
        return text_of(response)

    if reason == "refusal":
        # stop_details is populated ONLY for stop_reason == "refusal" - it is None for
        # end_turn, tool_use, max_tokens and the rest. Guard before reading it.
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None) if details else None
        print(f"  [iter {iteration}] refused (category: {category})")
        return text_of(response) or "[the model declined this request]"

    if reason == "pause_turn":
        # Unreachable in this lab - pause_turn is a *server* tool signal, emitted when
        # Anthropic's own sampling loop hits its iteration limit mid-turn. Both tools
        # here are client tools, so it never fires. The correct handling is to re-send
        # the same history to resume, with NO extra "continue" user message; see
        # 2-tools/tools.py for that branch in a loop that does use a server tool.
        print(f"  [iter {iteration}] pause_turn - unexpected without server tools")
        return text_of(response)

    print(f"  [iter {iteration}] stopped: {reason}")
    return text_of(response)


def run_agent(user_message: str) -> str:
    """Run the loop until Claude says it's done. Returns the final text.

    History lives in `messages` and is resent in full on every iteration - the Messages
    API is stateless.
    """
    messages: list[dict] = [{"role": "user", "content": user_message}]
    iteration = 0

    while True:
        iteration += 1

        # The safety cap. Not the exit condition - the exits are the stop_reason
        # branches below. Reaching here means something went wrong (a tool that always
        # errors, a prompt that sends Claude in circles), so say so loudly.
        if iteration > MAX_ITERATIONS:
            log.warning(
                "safety cap hit: %d iterations without end_turn. The loop should "
                "terminate on stop_reason, not here - check the tool results for a "
                "repeating failure.",
                MAX_ITERATIONS,
            )
            return "[stopped: safety iteration cap reached]"

        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )
        # No thinking / output_config.effort here - Haiku 4.5 supports neither and 400s.

        # Append the content blocks VERBATIM. Flattening to text here would drop the
        # tool_use blocks, and the next request would be rejected.
        messages.append({"role": "assistant", "content": response.content})
        render(response, iteration)

        # stop_reason is the authoritative signal. Branch on it by name - never on
        # "did the response contain a tool_use block", and never on the text.
        if response.stop_reason == "end_turn":
            print(f"  [iter {iteration}] end_turn - done after {iteration} iterations")
            return text_of(response)

        if response.stop_reason == "tool_use":
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                content, is_error = dispatch(block.name, block.input)
                preview = content if len(content) <= 160 else content[:157] + "..."
                print(f"    -> {'error: ' if is_error else ''}{preview}")

                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,  # must match the tool_use block
                        "content": content,
                        "is_error": is_error,
                    }
                )

            # Every result goes back in ONE user message. Splitting them across
            # messages teaches Claude to stop requesting tools in parallel.
            messages.append({"role": "user", "content": results})
            continue

        return handle_terminal_stop(response, iteration)


# ---------------------------------------------------------------------------
# 4. Entry point
# ---------------------------------------------------------------------------

DEMO = (
    "What was Dhaka's metro population in 2024, and what would it be after three more "
    "years at Bangladesh's current annual growth rate?"
)


def main() -> int:
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEMO
    print(f"you> {prompt}\n")

    try:
        answer = run_agent(prompt)
    except anthropic.APIError as exc:
        print(f"  [api error: {exc}]")
        return 1

    print(f"\nclaude> {answer}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
