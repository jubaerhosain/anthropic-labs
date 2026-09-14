# Multi-tool agent loop

A hand-written agentic loop whose control flow is `stop_reason` and nothing else, with two
client-side tools: a `calculator` and a `web_search` **stub**.

```bash
python3 ccar-f/agentic-loops/agent_loop.py                  # sequential demo
python3 ccar-f/agentic-loops/agent_loop.py "what is 7**12"  # your own prompt
```

## The exercise, mapped to the code

| Step | Where |
|---|---|
| 1. Two tools with proper JSON Schema `input_schema` | `CALCULATOR`, `WEB_SEARCH` -> `TOOLS` |
| 2. Loop that inspects `stop_reason` after each response | `run_agent()` - `while True`, branches on `response.stop_reason` |
| 3. Handle `tool_use`: execute, build `tool_result`, append | the `stop_reason == "tool_use"` branch |
| 4. Handle `end_turn`: extract and return the final text | the `stop_reason == "end_turn"` branch -> `text_of()` |
| 5. Sequential multi-tool prompt | `DEMO` - search, then calculate with what you found |
| 6. Safety cap with a warning | `MAX_ITERATIONS = 20` + `log.warning` at the top of the loop |

## Two tools, both client-side

Lab 2 (`2-tools/tools.py`) also has a `web_search`, but Anthropic's **server-side** one -
declared as `{"type": "web_search_20250305", "name": "web_search"}`, executed on
Anthropic's infrastructure, with results already inside the response we get back. The
`web_search` here is a plain client tool: `name` / `description` / `input_schema`, no
`type` field, and a function in this file that returns canned snippets.

The distinction is the whole reason the loop exists:

- A **client** tool stops the turn. The API returns `stop_reason: "tool_use"` and nothing
  continues until we run the function and send a `tool_result` back.
- A **server** tool never stops the turn. It runs mid-turn and we never see a `tool_use`
  block for it - only `server_tool_use` and its result, already resolved.

A loop that checks `stop_reason` handles both correctly without knowing which is which.

## Why not `for _ in range(20)`

Because then the cap *is* the control flow, and the loop ends because it ran out of turns
rather than because Claude decided it was finished. The two failure modes:

- The cap is too low and a legitimate long task is silently truncated mid-work.
- The cap is the only thing stopping a loop that's already broken, so nothing warns you.

`while True` inverts it. The exits are the `stop_reason` branches; `MAX_ITERATIONS` is a
backstop that should never fire, and logs a warning if it does. Normal queries here
terminate in 1-3 iterations.

`stop_reason` is also the *only* authoritative signal. Don't infer "Claude is done" from
the absence of a `tool_use` block, and never from the response text - a turn can contain
both text and a tool call (iteration 1 of the demo does exactly that).

## The stop reasons

| `stop_reason` | Meaning | Handling |
|---|---|---|
| `end_turn` | Claude is done | extract text, return - the normal exit |
| `tool_use` | Claude wants a client tool | run it, append the result, loop |
| `max_tokens` | response truncated | return partial text, raise `MAX_TOKENS` |
| `refusal` | safety classifier declined | `response.stop_details` is populated **only** here - guard before reading `.category` |
| `stop_sequence` | hit a configured stop sequence | return the text |
| `pause_turn` | a *server* tool hit its mid-turn limit | re-send the same history to resume; **never** append a "continue" user message. Unreachable in this lab - both tools are client-side |

## Message shapes that matter

Two ways to break the loop, both of which produce a 400 on the *next* request rather than
an obvious error where the mistake was:

```python
# Append the content blocks VERBATIM.
messages.append({"role": "assistant", "content": response.content})
```

Flattening to `text_of(response)` here drops the `tool_use` blocks, and the `tool_result`
you send next has nothing to attach to.

```python
# ALL results in ONE user message.
messages.append({"role": "user", "content": results})
```

Claude can request several tools in one turn (the demo's first iteration fires two
searches in parallel). Splitting the results across separate user messages trains it to
stop doing that. Each `tool_result` needs `tool_use_id` matching its `tool_use` block.

A failing tool returns a `tool_result` with `is_error: True`, never an exception -
`dispatch()` catches everything. Claude reads the error and can retry; a raised exception
just kills the loop.

## In production

You would not hand-write this. The SDK's tool runner drives the same loop:

```python
runner = client.beta.messages.tool_runner(
    model=MODEL, max_tokens=2048, tools=[...],
    messages=[{"role": "user", "content": prompt}],
)
for message in runner:
    ...
```

It handles the request -> execute -> append cycle, generates schemas from type hints via
`@beta_tool`, and stops when Claude stops asking for tools. It does **not** auto-resume
`pause_turn`, which is the one case you still handle yourself. The point of writing the
loop by hand once is knowing what it's doing for you.

## Cost

`claude-haiku-4-5` at $1 / $5 per million tokens; the demo is three turns, well under a
cent. The search stub makes no network call, so unlike lab 2 there is no per-search
charge. The Messages API is stateless - the whole history is resent every iteration, so
input tokens grow with each tool result.
