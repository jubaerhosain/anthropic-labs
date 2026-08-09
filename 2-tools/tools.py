"""Lab 2 - tool use, with the agentic loop written by hand.

Three tools are wired into one chat loop, and the point of the lab is the contrast
between them:

    calculator    we define it, we execute it   (client-side)
    get_weather   we define it, we execute it   (client-side)
    web_search    Anthropic defines it, Anthropic executes it   (server-side)

A client tool stops the turn: Claude emits a `tool_use` block, the API returns, and
nothing continues until we run the function and send a `tool_result` back. A server tool
never reaches us - it runs mid-turn on Anthropic's infrastructure and its results are
already inside the response we get.

Run it:

    python3 2-tools/tools.py                 # interactive REPL
    python3 2-tools/tools.py "what is 7**12" # one turn, then exit

Where to go next, once this loop makes sense:
  - the SDK's tool runner drives this loop for you: `@beta_tool` +
    `client.beta.messages.tool_runner(...)`
  - `strict: True` on a tool schema guarantees the input validates (needs
    `additionalProperties: false`)
  - streaming with tools: `client.messages.stream(...)` + `stream.get_final_message()`
  - human-in-the-loop: prompt for approval in `dispatch()` before a tool with real
    side effects runs
"""

import ast
import json
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# .env lives at the repo root, one level up - load it by path so this script works
# no matter which directory you run it from.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

if not os.getenv("ANTHROPIC_API_KEY"):
    raise RuntimeError(
        "ANTHROPIC_API_KEY not found. Copy .env.example to .env and paste your key in."
    )

# The client reads ANTHROPIC_API_KEY from the environment - don't pass it explicitly.
client = anthropic.Anthropic()

MODEL = "claude-haiku-4-5"  # cheapest model: $1 / $5 per million tokens
IN_PER_MTOK, OUT_PER_MTOK = 1.00, 5.00  # USD, claude-haiku-4-5
SEARCH_PER_1K = 10.00  # USD per 1,000 web searches - the one non-token charge here


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

GET_WEATHER = {
    "name": "get_weather",
    "description": (
        "Look up the current weather for a city. Call this when the user asks about "
        "weather or temperature anywhere. Note: this returns canned demo data from a "
        "small fixed table, not a live forecast - say so when you report it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name, e.g. 'Tokyo' or 'Dhaka'.",
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],  # enum-constrained parameter
                "description": "Temperature unit. Defaults to celsius.",
            },
        },
        "required": ["city"],
    },
}

# Server tool. We declare it and never implement it - Anthropic runs it.
# Version gotcha: `web_search_20260209` (with dynamic filtering) needs Opus 4.6+ or
# Sonnet 4.6+. Haiku 4.5 takes the basic version below; sending the newer one 400s.
# max_uses caps spend - searches bill separately from tokens.
WEB_SEARCH = {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}

TOOLS = [CALCULATOR, GET_WEATHER, WEB_SEARCH]


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


# Canned data. A real implementation would call a weather API here; the shape of the
# loop is identical either way.
_WEATHER = {
    "tokyo": (18, "light rain"),
    "dhaka": (31, "hazy sunshine"),
    "london": (11, "overcast"),
    "san francisco": (15, "fog"),
    "reykjavik": (2, "snow showers"),
}


def get_weather(city: str, unit: str = "celsius") -> str:
    """Return demo weather for a city, in celsius or fahrenheit."""
    celsius, sky = _WEATHER.get(city.strip().lower(), (21, "clear"))

    if unit == "fahrenheit":
        return f"{city}: {round(celsius * 9 / 5 + 32)}F, {sky} (demo data)"
    return f"{city}: {celsius}C, {sky} (demo data)"


def dispatch(name: str, tool_input: dict) -> tuple[str, bool]:
    """Run a client-side tool. Returns (content, is_error).

    Failures come back as an error *result*, never an exception. Claude sees the
    message and can retry with different input; raising here would kill the loop.
    """
    try:
        if name == "calculator":
            return calculator(**tool_input), False
        if name == "get_weather":
            return get_weather(**tool_input), False
        return f"No tool named {name!r}.", True
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}", True


# ---------------------------------------------------------------------------
# 3. The agent
# ---------------------------------------------------------------------------

SYSTEM = (
    "You are a hands-on assistant with three tools. Use the calculator for any exact "
    "arithmetic, get_weather for weather questions, and web search for anything that "
    "depends on current information. Answer briefly."
)


class ToolBot:
    """A multi-turn conversation with Claude that can call tools.

    chat() runs the agentic loop until Claude stops asking for tools, printing a trace
    of every block along the way. History lives in .messages and is resent every turn.
    """

    def __init__(
        self,
        system: str | None = SYSTEM,
        model: str = MODEL,
        max_tokens: int = 2048,
        max_iterations: int = 8,
    ):
        self.system = system
        self.model = model
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations  # a bad tool result can loop forever
        self.messages: list[dict] = []
        self.last_reply: str = ""
        self.input_tokens = 0
        self.output_tokens = 0
        self.searches = 0

    def chat(self, user_message: str) -> None:
        self.messages.append({"role": "user", "content": user_message})
        response = None

        for _ in range(self.max_iterations):
            params = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "tools": TOOLS,
                "messages": self.messages,
            }
            if self.system:
                params["system"] = self.system
            # No output_config/effort here - Haiku 4.5 has no effort parameter and 400s.

            response = client.messages.create(**params)
            self._account(response)

            # Append the content blocks verbatim. Flattening to text here would drop
            # the tool_use blocks, and the next request would be rejected.
            self.messages.append({"role": "assistant", "content": response.content})
            self._render(response)

            if response.stop_reason == "max_tokens":
                print("[truncated - raise max_tokens]")
                break

            if response.stop_reason == "pause_turn":
                # A server tool hit its iteration cap mid-turn. Re-send the same
                # history to resume - do NOT append a 'continue' user message.
                continue

            if response.stop_reason != "tool_use":
                break  # end_turn, refusal, stop_sequence: nothing left to run

            # Run every requested tool, then send ALL results back in ONE user
            # message. Splitting them across messages teaches Claude to stop
            # requesting tools in parallel.
            results = []
            for block in response.content:
                # Exactly "tool_use" - server_tool_use blocks are Anthropic's to run.
                if block.type != "tool_use":
                    continue

                content, is_error = dispatch(block.name, block.input)
                print(f"  [-> {'error: ' if is_error else ''}{content}]")
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,  # must match the tool_use block
                        "content": content,
                        "is_error": is_error,
                    }
                )

            self.messages.append({"role": "user", "content": results})
        else:
            print(f"[stopped after {self.max_iterations} iterations]")

        self.last_reply = "".join(
            b.text for b in (response.content if response else []) if b.type == "text"
        )

    def _render(self, response) -> None:
        """Print one line per content block so the loop is observable."""
        for block in response.content:
            if block.type == "text":
                if block.text.strip():
                    print(block.text)
            elif block.type == "tool_use":
                print(f"  [tool: {block.name}({json.dumps(block.input)})]")
            elif block.type == "server_tool_use":
                print(f"  [searching: {block.input.get('query', '')!r}]")
            elif block.type == "web_search_tool_result":
                # Server tools return HTTP 200 even when they fail: .content is a list
                # of results on success and a single error object on failure.
                if isinstance(block.content, list):
                    print(f"  [search: {len(block.content)} results]")
                else:
                    print(f"  [search failed: {block.content.error_code}]")

    def _account(self, response) -> None:
        """Accumulate usage across every iteration, not just the last one."""
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens

        server = getattr(response.usage, "server_tool_use", None)  # absent if no search ran
        if server:
            self.searches += server.web_search_requests or 0

    def reset(self) -> None:
        """Forget the conversation. Keeps the system prompt and settings."""
        self.messages.clear()
        self.last_reply = ""

    def usage(self) -> dict:
        cost = (
            self.input_tokens * IN_PER_MTOK + self.output_tokens * OUT_PER_MTOK
        ) / 1e6 + self.searches * SEARCH_PER_1K / 1000
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "searches": self.searches,
            "cost_usd": cost,
        }


# ---------------------------------------------------------------------------
# 4. The chat loop
# ---------------------------------------------------------------------------


def print_usage(bot: ToolBot) -> None:
    u = bot.usage()
    print(
        f"  {u['input_tokens']:>7} in  {u['output_tokens']:>6} out  "
        f"{u['searches']:>2} searches  ~${u['cost_usd']:.5f}"
    )


def repl(bot: ToolBot) -> None:
    print(f"Lab 2 - tool use on {MODEL}. Commands: /reset  /usage  /quit\n")

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

        try:
            bot.chat(user_message)
        except anthropic.APIError as exc:
            print(f"  [api error: {exc}]")

    print_usage(bot)


if __name__ == "__main__":
    bot = ToolBot()

    if len(sys.argv) > 1:  # one-shot mode: handy for scripted checks
        bot.chat(" ".join(sys.argv[1:]))
        print_usage(bot)
    else:
        repl(bot)
