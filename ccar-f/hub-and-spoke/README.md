# Hub-and-spoke research coordinator

One coordinator (the hub) owns decomposition, delegation, aggregation and refinement. Two
subagents (the spokes) do the research in isolation and never talk to each other.

```bash
python3 ccar-f/hub-and-spoke/coordinator.py                        # renewable energy demo
python3 ccar-f/hub-and-spoke/coordinator.py "urban water security" # your own topic
python3 ccar-f/hub-and-spoke/coordinator.py --report out.md        # save the report
python3 ccar-f/hub-and-spoke/coordinator.py --no-brief             # unaided decomposition
python3 ccar-f/hub-and-spoke/coordinator.py "AI safety" --must-cover "interpretability,evals"
```

```
                    topic
                      |
                      v
                [ COORDINATOR ]  <- every arrow starts or ends here
                  |         ^
          +-------+         +-------+
          |                         |
          v                         v
    web_researcher            document_analyst
    (search stub)             (docs/ corpus)
```

## Exercise checklist

- [x] **1. Coordinator agent taking a broad topic** — `research()`, with `COORDINATOR_SYSTEM`
      defining the hub's four responsibilities. Returns the report plus an audit trail.
- [x] **2. Decomposition into 5+ subtopics** — `decompose()`: enumerate, breadth-audit,
      consolidate. `MIN_SUBTOPICS = 5` is a floor it refuses to fall below; runs produce
      11-14. For renewable energy it covers solar, wind, geothermal, tidal, biomass and
      fusion.
- [x] **3. Two spokes with explicit context passing** — `run_spoke()` and `ContextPacket`,
      which carries the research goal, the assignment, why this spoke got it, prior agents'
      findings, known gaps and targeted queries.
- [x] **4. Aggregation and coverage evaluation** — `aggregate()` scores every subtopic
      covered / partial / missing with evidence; `render_coverage()` prints the table.
- [x] **5. Iterative refinement loop** — the `while` loop in `research()`: evaluate, plan
      targeted follow-ups with `plan_followups()`, re-delegate, re-evaluate, until the
      threshold is met or `MAX_REFINEMENT_ROUNDS` is reached.
- [x] **6. Tested on renewable energy technologies** — six required subjects verified in the
      finished report by `verify_required()`. Last run: **6/6 required subjects, each with
      its own section**, overall subtopic coverage 68% → 79% → 82% across three rounds.

## What each piece is for

| Concern | Where |
|---|---|
| Hub's role and responsibilities | `COORDINATOR_SYSTEM` |
| Task decomposition | `decompose()` — three passes |
| Breadth floor / ceiling | `MIN_SUBTOPICS`, `MAX_SUBTOPICS` |
| Scope mandated by the brief | `apply_scope_requirements()` |
| Explicit context passing | `ContextPacket.render()` |
| Subagent isolation | `run_spoke()` — a fresh `messages` list per call |
| Aggregation and grading | `aggregate()`, `coverage_score()` |
| Re-delegation planning | `plan_followups()` |
| The refinement loop | `research()` |
| Final verification | `verify_required()` |

## Isolation is structural here, not a convention

`run_spoke()` builds a brand-new `messages` list on every call. A spoke cannot see the
coordinator's history, the other spoke's history, or its own history from a previous round.
Everything it knows arrived in one string.

That is why `ContextPacket` is a dataclass with named fields rather than an f-string. Drop
the research goal from a prompt and nothing raises — you just get a subagent optimising for
the wrong objective and a report you cannot explain. The failure is silent, so the guard has
to be structural.

The direct consequence, and the diagnostic the whole design is built around: **a thin spoke
report is almost always a hub bug.** Before concluding a subagent is weak, read the packet
it was sent. Round 1 prints the size of each one.

## Narrow decomposition, and why the fix is three passes

The classic failure is a report on renewable energy covering solar and wind and nothing
else. The bug is not in the subagents. A spoke assigned "solar" returns excellent work on
solar and has no way to know geothermal was ever in scope — nothing in the system will
report the absence, because nothing was ever asked to look.

So `decompose()` runs:

1. **Enumerate.** The obvious list. On its own it reliably returns the prominent members of
   a category and stops.
2. **Breadth-audit.** Show the model its own list and ask what a knowledgeable reviewer
   would say was missing. The model always knew geothermal existed; nothing had asked it to
   audit its own output for omissions. This pass typically adds 3-12 subtopics.
3. **Consolidate.** The audit deliberately over-generates, so merge back to a delegable
   number — by combining, never by dropping.

Three rules that were each added after watching a run fail without them:

- **No umbrella subtopics.** "Emerging technologies" is not a subtopic, it is the absence of
  one: a spoke handed it has nothing to research. Applied to all three passes — an early
  version banned it in the audit only, and consolidation promptly swallowed "Nuclear fusion"
  back into an "Emerging technologies" bucket.
- **The audit adds peers, not sub-variants.** One run split storage into four sub-subtopics,
  which added no breadth and forced consolidation to merge genuinely distinct technologies
  into joint buckets like "Solar photovoltaic and thermal" — which then graded partial
  forever, because each half needed full documentation.
- **Consolidation must be lossless, and that is checked in code.** `_is_dropped()` verifies
  every pre-merge subtopic still appears by name; anything that vanished is restored as its
  own subtopic, exceeding the cap if necessary. The cap yields to breadth because the two
  errors are not symmetric: an over-broad decomposition costs tokens and is fixable at any
  point, while a dropped category is unrecoverable — no spoke is told to look for it and no
  coverage check is ever scored against it.

## The seeded gap

Both corpora are rigged so the refinement loop has honest work to do:

- `docs/05-marine-energy-placeholder.txt` is a placeholder memo that explicitly defers
  marine energy and says external sources are required.
- `docs/06-frontier-technologies.txt` lists fusion as tracking-only with no assessment.
- In the search stub, corpus entries are keyed `(subject, qualifiers)`. Most subjects have
  no qualifiers and match on the category name. Tidal and fusion do have them, so
  `"tidal energy"` and `"fusion energy"` return nothing while
  `"tidal stream turbine capacity factor"` and `"fusion tokamak net energy gain"` succeed.

A broad first sweep therefore comes back genuinely thin on exactly those two. The hub
notices, `plan_followups()` writes queries naming specific technologies and projects, and
round two fills them. Round 1 coverage lands around 50-68%; three rounds reach 82%+.

## Coverage must be monotonic

`aggregate()` never lets a verdict move backwards, and does not re-grade settled subtopics.

This is load-bearing. An LLM grader re-scoring the same growing pile of evidence does not
return the same verdict twice, so a subtopic judged covered in round 1 comes back partial in
round 2 on wording alone. The loop then chases a number that moves in both directions and
never converges — an earlier version of this code went **75% → 60% while strictly gaining
evidence**. Since `findings` is append-only, no later round can make an earlier round's
evidence insufficient, so a verdict may only improve.

The other grader failure worth knowing: it will grade against one source unless told not to.
Early runs marked subtopics missing because the internal corpus was silent, while the web
spoke had already supplied the figures — the two spokes working exactly as designed, scored
as a shortfall. Coverage froze at 30% and re-delegated work that was already done. "Not in
corpus" is an answer about the corpus, never about the subtopic.

## A coordinator is not a dispatcher

A dispatcher delegates once and concatenates. The hub:

- scores its own aggregate output per subtopic and is told to name the missing fact before
  writing "partial",
- plans *new, more specific* queries against the gaps — re-sending the same assignment
  returns the same empty result,
- chases at most `MAX_GAPS_PER_ROUND = 5` gaps, worst first, because re-delegating every
  shortfall at once rebuilds the round-1 prompt that already failed,
- runs the spokes **sequentially**, so the document analyst's packet can carry the web
  researcher's findings. The spokes have no channel between them; anything that crosses does
  so because the hub carried it.

## `must_cover`: scope the taxonomy will not infer

Fusion is not a renewable technology — its fuel is mined lithium and deuterium — so a
correct decomposition omits it about half the time, and arguing taxonomy in the prompt does
not reliably change that. `--must-cover` lets the brief assert scope directly. Anything
injected is announced (`injected by brief -> fusion`), so the report's breadth is never
mistaken for something the decomposition found on its own. Run `--no-brief` to see the
unaided version.

This is a coordinator input rather than a prompt tweak on purpose: a requirement stated here
is auditable and always applied; the same requirement hinted at in a prompt is neither.

## Failure handling

| Failure | Handling |
|---|---|
| Spoke exhausts its iteration budget | One more turn with no tools, forcing it to write up what it gathered. Returning scraps would make the next coverage check read as a gap it never addressed, and the hub would re-delegate work that was already done. |
| Forced tool call truncated at `max_tokens` | `structured_call()` raises where the cause is obvious, instead of a `KeyError` three frames away. |
| Coverage evaluation or follow-up planning fails | Logged; the loop stops and synthesises from `findings`. Losing a whole run over one bookkeeping step would throw away good research. |
| Tool raises | `dispatch()` returns `is_error: True`; the spoke reads it and retries. |
| Threshold never met | Warns with the open gaps and reports the real number. |

## Cost

`claude-haiku-4-5` at $1 / $5 per million tokens. A full run is roughly 25-30 API calls —
two spokes × three rounds (each spoke 3-8 iterations), plus decomposition, grading,
follow-up planning and synthesis. The search stub makes no network calls. Well under a cent.
