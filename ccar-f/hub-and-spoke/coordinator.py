"""Hub-and-spoke research coordinator - one hub, two isolated spokes, a refinement loop.

    topic
      |
      v
  [ COORDINATOR ]  <- the hub. Owns decomposition, delegation, aggregation, refinement.
      |      ^         Every arrow starts or ends here. The spokes never touch each other.
      |      |
      +------+------+
      |             |
      v             v
  web_researcher  document_analyst        <- the spokes. Isolated: separate `messages`
  (search stub)   (docs/ corpus)             lists, separate system prompts, no shared
                                             memory, no inherited context.

Three things this file is trying to make impossible to miss:

1. SUBAGENT ISOLATION IS REAL HERE, not a comment. `run_spoke()` builds a brand-new
   `messages` list per call and never sees the coordinator's history. Whatever the hub
   forgets to write into the prompt genuinely does not exist for the spoke. That is why
   `ContextPacket` exists: it is the *only* channel, so it is a dataclass with named
   fields rather than an f-string someone can quietly drop a field from.

2. BREADTH IS THE HUB'S JOB. The classic failure - "renewable energy" decomposed into
   solar and wind, report comes back missing four categories - is a *decomposition* bug.
   It is not a spoke bug. A spoke handed "solar" returns excellent work on solar; it has
   no way to know geothermal was supposed to be in scope. So `decompose()` runs a second
   breadth-audit pass against its own first answer, and enforces a MIN_SUBTOPICS floor.

3. A COORDINATOR IS NOT A DISPATCHER. A dispatcher delegates once and concatenates. The
   hub here scores its own aggregate output per subtopic, plans *targeted* follow-up
   queries for whatever came back thin, and re-delegates until coverage clears the
   threshold or it runs out of rounds. `refine()` is the loop.

The corpora are seeded so the loop has honest work to do: the document set is a
placeholder memo for marine energy and a tracking-only line for fusion, and the search
stub's tidal/fusion entries need specific technical terms to match. A broad first-round
sweep genuinely comes back with those two thin; round two's targeted queries fill them.

Run it:

    python3 ccar-f/hub-and-spoke/coordinator.py                       # renewable energy
    python3 ccar-f/hub-and-spoke/coordinator.py "urban water security"
    python3 ccar-f/hub-and-spoke/coordinator.py --report out.md       # save the report
"""

import json
import logging
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

client = anthropic.Anthropic()

MODEL = "claude-haiku-4-5"
MAX_TOKENS = 4096

DOCS_DIR = Path(__file__).resolve().parent / "docs"

# Decomposition floor. Fewer than this and the hub has under-decomposed - the exact bug
# that produces a solar-and-wind-only report. `decompose()` retries rather than proceed.
MIN_SUBTOPICS = 5

# Ceiling on what actually gets delegated. Breadth is found first and consolidated after,
# never traded away up front: an over-broad decomposition is a cost and focus problem with
# an easy fix (merge related items), while a narrow one silently loses whole categories and
# has no fix at all downstream. So `decompose()` over-generates, then merges down to this.
MAX_SUBTOPICS = 12

# Refinement stops when weighted coverage reaches this. 1.0 means every subtopic must be
# "covered" - "partial" scores 0.5 and will not clear the bar on its own.
COVERAGE_THRESHOLD = 1.0
MAX_REFINEMENT_ROUNDS = 3

# Gaps chased per refinement round, worst first. Re-delegating every shortfall at once
# rebuilds the round-1 prompt that already failed, and the packets grow until the spoke
# runs out of iterations before it runs out of assignment. Fewer gaps, sharper prompts.
MAX_GAPS_PER_ROUND = 5

# Safety cap inside a spoke's own tool loop. Like agent_loop.py, this is a backstop and
# not the exit condition - spokes exit on stop_reason == "end_turn".
MAX_SPOKE_ITERATIONS = 12

# How much prior-agent context the hub forwards into the next spoke prompt. Bounded so
# prompts stay finite across rounds; the truncation is announced in the packet.
DIGEST_LIMIT = 2600

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)
log = logging.getLogger("coordinator")


# ---------------------------------------------------------------------------
# 1. Spoke tools - what each spoke can reach, and nothing more
# ---------------------------------------------------------------------------
# The two spokes differ only in their tools and system prompt. That is the whole point
# of using two: they have genuinely different evidence available, so the hub needs both
# and has something real to aggregate rather than two paraphrases of one source.

WEB_SEARCH = {
    "name": "web_search",
    "description": (
        "Search the web for published figures, costs, capacities and project milestones. "
        "Returns short snippets. Prefer several narrow, technical queries over one broad "
        "one - a query naming a specific technology, metric, project or milestone returns "
        "far more than a generic category name."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query, e.g. 'tidal stream turbine capacity factor'.",
            },
            "max_results": {
                "type": "integer",
                "description": "How many snippets to return. Defaults to 4.",
            },
        },
        "required": ["query"],
    },
}

LIST_DOCUMENTS = {
    "name": "list_documents",
    "description": (
        "List the filenames in the internal document corpus. Call this first so you know "
        "what exists before reading anything."
    ),
    "input_schema": {"type": "object", "properties": {}},
}

READ_DOCUMENT = {
    "name": "read_document",
    "description": (
        "Read one document from the internal corpus by exact filename, as listed by "
        "list_documents. Returns the full text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Exact filename, e.g. '03-geothermal.txt'.",
            }
        },
        "required": ["filename"],
    },
}

# Canned corpus. Entries are (subject, qualifiers): a query matches when the subject word
# appears AND - if qualifiers are listed - at least one qualifier does too.
#
# That qualifier gate is the seeded gap. Solar, wind, geothermal, biomass and hydro have
# no qualifiers, so a broad first-round query finds them. Tidal and fusion do, so
# "tidal energy" returns nothing and only a targeted follow-up naming a turbine type, a
# metric or a named project gets through. The hub has to notice the hole and ask again.
_CORPUS: dict[tuple[str, tuple[str, ...]], list[str]] = {
    ("solar", ()): [
        "Utility-scale solar PV reached a global weighted-average LCOE of $0.043/kWh in "
        "2024, roughly a 90% decline since 2010.",
        "Global solar PV additions were approximately 599 GW in 2024, about three-quarters "
        "of all new renewable capacity worldwide.",
        "Perovskite-silicon tandem cells hit 33.9% certified efficiency in 2024; damp-heat "
        "durability, not efficiency, is the barrier to commercialisation.",
    ],
    ("wind", ()): [
        "Onshore wind LCOE averaged $0.034/kWh in 2024, the cheapest new-build generation "
        "in most markets.",
        "Global wind additions were about 117 GW in 2024, of which roughly 8 GW offshore.",
        "Floating offshore wind cumulative capacity remains near 0.3 GW, with costs 2-3x "
        "fixed-bottom offshore and convergence expected through the 2030s.",
    ],
    ("geothermal", ()): [
        "Global geothermal capacity is about 16 GW, with capacity factors of 70-90% - the "
        "highest of any renewable source.",
        "Fervo Energy's Cape Station EGS project demonstrated 3.5 MW per well flow rates in "
        "2024, with drilling costs falling as horizontal drilling practice transfers in.",
        "About 175 GWth of direct-use geothermal heat is installed globally, exceeding the "
        "geothermal power sector and usually excluded from renewable electricity figures.",
    ],
    ("biomass", ()): [
        "Modern bioenergy supplies roughly 55% of global renewable final energy when "
        "traditional biomass is included - more than wind and solar combined.",
        "Carbon payback for woody biomass ranges from under 5 years for agricultural "
        "residues to 40-100 years for whole mature trees, which is the core of the "
        "carbon-neutrality dispute.",
        "Stockholm Exergi took FID on an 800,000 tonne/year BECCS project in 2024.",
    ],
    ("bioenergy", ()): [
        "European biomethane production reached approximately 4.9 bcm in 2024, with "
        "manure-derived biogas often net-negative once avoided methane is credited.",
    ],
    # "hydro" as a subject would also match "hydrogen" - subjects are matched as
    # substrings - so the two spellings are listed explicitly instead.
    ("hydroelectric", ()): [
        "Hydropower remains the largest renewable electricity source at roughly 1,400 GW "
        "installed, generating about 4,300 TWh annually.",
        "Run-of-river schemes trade storage for lower ecological impact, typically running "
        "at 40-60% capacity factor against 30-50% for reservoir plant.",
        "Pumped storage is 95% of global installed storage energy at roughly 180 GW, with "
        "70-80% round-trip efficiency. Drought cut Northern Hemisphere hydro output "
        "materially in 2023, and climate projections make reservoir yield less predictable.",
    ],
    ("hydropower", ()): [
        "Hydropower's roughly 1,400 GW makes it the largest renewable electricity source; "
        "growth is limited by remaining sites, ecological objection and drought risk.",
    ],
    ("csp", ()): [
        "Concentrated solar power has about 6.8 GW installed globally, an order of "
        "magnitude behind PV, at an LCOE near $0.118/kWh - roughly triple utility PV.",
        "CSP's advantage is integrated thermal storage: molten-salt towers such as Noor "
        "Ouarzazate III (150 MW, 7 hours) and Cerro Dominador in Chile dispatch into the "
        "evening peak, reaching 40-70% capacity factors that PV cannot match without "
        "batteries.",
        "Parabolic trough remains the most-deployed CSP configuration; central-receiver "
        "towers reach higher temperatures and better cycle efficiency. Solar thermal for "
        "heat - water heating and industrial process heat - is a far larger market than "
        "CSP power, at roughly 560 GWth installed worldwide.",
    ],
    ("concentrat", ()): [
        "Concentrated solar power totals about 6.8 GW globally at roughly $0.118/kWh, "
        "competing on dispatchability through molten-salt thermal storage rather than on "
        "cost per kWh.",
    ],
    ("waste", ("energy", "municipal", "incineration", "solid", "msw", "heat", "thermal",
               "gasification", "landfill", "capacity", "plant")): [
        "Waste-to-energy incineration has roughly 100 GW of thermal capacity globally, "
        "concentrated in Japan, China and northern Europe; Denmark's Amager Bakke is the "
        "reference modern plant.",
        "Electrical efficiency is low at 20-25%, rising above 80% total utilisation where "
        "district heating takes the waste heat. Roughly half the carbon in municipal solid "
        "waste is fossil-derived plastic, so only the biogenic fraction counts as "
        "renewable - and landfill-methane avoidance often dominates the net balance.",
    ],
    ("nuclear", ("fission", "reactor", "smr", "modular", "lcoe", "capacity", "factor",
                 "construction", "cost", "deployment", "fleet")): [
        "Nuclear fission supplies about 9% of world electricity from roughly 370 GW, at "
        "capacity factors above 90% - the highest of any generation source.",
        "New-build LCOE is $0.14-0.22/kWh in Western markets, driven by construction "
        "overruns: Vogtle 3-4 and Hinkley Point C both roughly doubled their budgets. "
        "Small modular reactors promise factory serialisation, but NuScale's Idaho project "
        "was cancelled in 2023 on cost, and no Western SMR is yet operating commercially.",
    ],
    ("otec", ()): [
        "Ocean thermal energy conversion exploits a 20 C surface-to-deep gradient at 3-5% "
        "theoretical efficiency; only pilot plant exists, such as Makai's 100 kW unit in "
        "Hawaii. Cold-water pipe engineering at commercial scale is the unsolved problem.",
    ],
    ("microgrid", ()): [
        "Microgrids pair distributed generation with storage and islanding control; "
        "distributed solar is roughly 40% of global PV capacity, and virtual power plants "
        "aggregating home batteries now bid into wholesale markets in Australia, "
        "California and Germany.",
    ],
    ("distributed", ()): [
        "Distributed generation - rooftop PV, small wind, behind-the-meter batteries - is "
        "about 40% of global solar capacity and shifts the grid problem from generation "
        "adequacy to distribution-network hosting capacity and reverse power flow.",
    ],
    ("storage", ()): [
        "Lithium iron phosphate pack prices fell below $100/kWh in 2024; four-hour duration "
        "is the commercial grid-storage standard.",
        "Form Energy's 100-hour iron-air chemistry began first installations in 2024-2025, "
        "targeting the unsolved 10-100 hour long-duration regime.",
    ],
    # --- seeded gap: needs a specific query, not the category name -------------------
    ("tidal", ("stream", "turbine", "capacity", "barrage", "lagoon", "range", "project",
               "cost", "lcoe", "deployment", "meygen", "sihwa", "orbital", "marine",
               "ocean", "technical")): [
        "MeyGen in the Pentland Firth is the largest tidal stream array operating, at 6 MW "
        "installed with over 50 GWh delivered cumulatively by 2024.",
        "Tidal stream capacity factors run 30-40% and - uniquely among renewables - output "
        "is deterministically predictable years ahead from lunar and solar ephemerides.",
        "Orbital Marine's O2 floating turbine is rated 2 MW. Tidal stream LCOE is currently "
        "£150-250/MWh, with the UK Contracts for Difference ringfence pricing 2024 awards "
        "at about £172/MWh and a target below £100/MWh at 1 GW of deployment.",
        "Tidal range schemes are a separate technology: the 254 MW Sihwa Lake barrage in "
        "South Korea and the 240 MW La Rance plant in France, operating since 1966, are the "
        "two largest. Estuarine ecological impact is the binding constraint on new barrages.",
    ],
    # Aliases: a decomposition may name this subtopic "marine energy" or "ocean energy"
    # rather than "tidal", and a spoke searching its own subtopic name must be able to
    # reach the evidence. Same qualifier gate, so the seeded gap survives either phrasing.
    ("marine", ("tidal", "stream", "turbine", "capacity", "barrage", "lagoon", "project",
                "cost", "lcoe", "deployment", "technical", "assessment", "installed")): [
        "Tidal stream and tidal range are the two marine technologies with operating "
        "utility-scale plant: MeyGen (6 MW, Pentland Firth) and the 254 MW Sihwa Lake and "
        "240 MW La Rance barrages respectively.",
        "Marine energy capacity factors of 30-40% come with output predictable years ahead "
        "from lunar ephemerides; tidal stream LCOE is £150-250/MWh, targeting sub-£100 at "
        "1 GW deployed.",
    ],
    ("ocean", ("tidal", "stream", "turbine", "capacity", "thermal", "otec", "project",
               "cost", "lcoe", "deployment", "technical", "assessment", "installed")): [
        "Total global ocean energy capacity remains under 1 GW, dominated by two tidal "
        "range barrages built decades apart; ocean thermal energy conversion (OTEC) has "
        "only pilot-scale plant.",
    ],
    ("wave", ("energy", "converter", "device", "power", "resource", "marine", "ocean")): [
        "Wave energy remains pre-commercial with no convergent device architecture; global "
        "installed capacity is a few MW across competing point-absorber, oscillating water "
        "column and attenuator designs.",
    ],
    ("fusion", ("tokamak", "stellarator", "ignition", "gain", "net", "iter", "nif",
                "milestone", "timeline", "private", "funding", "confinement", "commonwealth",
                "sparc", "helion", "magnet", "frontier", "pre-commercial", "experimental")): [
        "The National Ignition Facility achieved scientific breakeven in December 2022 "
        "(3.15 MJ out from 2.05 MJ of laser energy in) and has since repeated it with "
        "higher gain, though facility wall-plug energy remains far above output.",
        "Private fusion funding passed $7 billion cumulatively by 2024 across roughly 45 "
        "companies. Commonwealth Fusion Systems' SPARC tokamak targets Q>1 around 2027, "
        "enabled by REBCO high-temperature superconducting magnets demonstrated at 20 tesla.",
        "ITER's revised baseline pushes first plasma to 2034 and deuterium-tritium operation "
        "to 2039. Tritium breeding self-sufficiency and neutron-resistant first-wall "
        "materials are the two unsolved engineering problems common to all magnetic designs.",
        "Fusion sits outside the conventional renewable taxonomy - its fuel is mined lithium "
        "and deuterium - but it is routinely grouped with renewables in frontier clean-firm "
        "power assessments, so its exclusion from a breadth review reads as an oversight.",
    ],
    ("hydrogen", ()): [
        "Green hydrogen costs $4-6/kg against $1-2/kg for unabated steam methane reforming. "
        "Alkaline electrolysers run $500-1,400/kW installed and PEM $700-1,800/kW, at "
        "60-70% LHV efficiency.",
        "Announced project pipelines exceed 400 GW of electrolysis while under 2 GW has "
        "reached FID - the gap between announcement and offtake is the sector's defining "
        "feature.",
        "Round-trip efficiency through electrolysis, storage and a fuel cell is only "
        "30-40%, which is why hydrogen loses to batteries for short-duration storage and "
        "is argued for instead in steel, ammonia, shipping and aviation, where direct "
        "electrification has no route.",
    ],
    ("electrolyser", ()): [
        "Alkaline electrolysers cost $500-1,400/kW installed, PEM $700-1,800/kW, both at "
        "60-70% LHV efficiency; stack lifetime and load-following ability differ more than "
        "headline cost does.",
    ],
    ("cell", ("fuel", "cells", "pem", "sofc", "efficiency", "stack")): [
        "PEM fuel cells convert hydrogen to electricity at 50-60% efficiency; solid-oxide "
        "reaches 60% electrical and above 85% in combined heat and power.",
    ],
}

_STUB_PREFIX = "[stub search results - canned corpus, no network call]"


def web_search(query: str, max_results: int = 4) -> str:
    """Canned snippet lookup. Subject word required; one qualifier too, where listed."""
    words = {w.strip(".,;:?!'\"()[]").lower() for w in query.replace("-", " ").split()}

    snippets: list[str] = []
    for (subject, qualifiers), entries in _CORPUS.items():
        if not any(subject in w for w in words):
            continue
        if qualifiers and not any(q in words for q in qualifiers):
            continue
        # Aliases overlap with primary entries, so dedupe rather than repeat a snippet.
        snippets.extend(e for e in entries if e not in snippets)

    if not snippets:
        return (
            f"{_STUB_PREFIX}\nNo results for {query!r}. If this is a category name, retry "
            "with a specific technology, metric, named project or milestone."
        )

    numbered = "\n".join(
        f"{i}. {s}" for i, s in enumerate(snippets[: max(1, max_results)], start=1)
    )
    return f"{_STUB_PREFIX}\n{numbered}"


def list_documents() -> str:
    """Filenames in the corpus, so the analyst reads by name instead of guessing."""
    names = sorted(p.name for p in DOCS_DIR.glob("*.txt"))
    if not names:
        return f"No documents found in {DOCS_DIR}."
    return "Documents available:\n" + "\n".join(f"- {n}" for n in names)


def read_document(filename: str) -> str:
    """Read one corpus file. Path traversal is rejected - tool input is model output."""
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise ValueError("filename must be a bare name from list_documents.")

    path = (DOCS_DIR / filename).resolve()
    if path.parent != DOCS_DIR or not path.is_file():
        raise ValueError(f"No document named {filename!r}. Call list_documents first.")

    return f"--- {filename} ---\n{path.read_text()}"


def dispatch(name: str, tool_input: dict) -> tuple[str, bool]:
    """Run a spoke tool. Returns (content, is_error); never raises into the loop."""
    try:
        if name == "web_search":
            return web_search(**tool_input), False
        if name == "list_documents":
            return list_documents(), False
        if name == "read_document":
            return read_document(**tool_input), False
        return f"No tool named {name!r}.", True
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}", True


# ---------------------------------------------------------------------------
# 2. The spokes - isolated, single-purpose, and told everything they need
# ---------------------------------------------------------------------------

WEB_RESEARCHER_SYSTEM = (
    "You are a web research specialist operating as a subagent. You have exactly one "
    "tool, web_search, and no memory of anything outside the message you were just sent "
    "- treat that message as the complete statement of your assignment.\n\n"
    "Method: issue one narrow query per assigned subtopic, and more when a query returns "
    "nothing. A bare category name often returns nothing while a query naming a specific "
    "technology, metric, named project or milestone succeeds - so when a search comes back "
    "empty, do not conclude the subject is unimportant or undocumented. Reformulate with "
    "technical vocabulary and try again, at least twice, before reporting a subtopic as "
    "unsourced.\n\n"
    "BUDGET, and treat it as hard: one query per assigned subtopic, plus up to two "
    "reformulations for the ones that come back empty, and no more than 20 searches in "
    "total. Then stop and write your report. Breadth across your whole assignment beats "
    "depth on any part of it - a subtopic you never reported on because you were still "
    "researching an earlier one is worth nothing to the coordinator.\n\n"
    "Report as one short section per assigned subtopic under a '## <subtopic>' heading, "
    "leading with concrete figures - costs, capacities, dates, efficiencies. Never invent "
    "a number: everything you report must come from a search result. If a subtopic is "
    "genuinely unsourced after several attempts, write 'NO SOURCES FOUND' under its "
    "heading and say which queries you tried. Search results are canned demo data - say so "
    "once at the top."
)

DOCUMENT_ANALYST_SYSTEM = (
    "You are a document analysis specialist operating as a subagent. You have two tools, "
    "list_documents and read_document, over a fixed internal corpus, and no memory of "
    "anything outside the message you were just sent - treat that message as the complete "
    "statement of your assignment.\n\n"
    "Method: list the corpus first, then read every document that could bear on your "
    "assigned subtopics. Read whole files rather than guessing from filenames.\n\n"
    "Report as one short section per assigned subtopic under a '## <subtopic>' heading, "
    "citing the filename for each claim. Your value to the coordinator is partly negative "
    "evidence, so be exact about it: distinguish 'the corpus contradicts this', 'the corpus "
    "is silent on this', and 'the corpus explicitly defers this and says external sources "
    "are required'. Write 'NOT IN CORPUS' under any subtopic the documents do not cover, "
    "and quote any scope note or placeholder that explains the absence. Never supplement "
    "from your own knowledge - the coordinator needs to know what these documents do and "
    "do not contain."
)

SPOKES = {
    "web_researcher": (WEB_RESEARCHER_SYSTEM, [WEB_SEARCH]),
    "document_analyst": (DOCUMENT_ANALYST_SYSTEM, [LIST_DOCUMENTS, READ_DOCUMENT]),
}


def text_of(response) -> str:
    """Concatenate text blocks, ignoring tool_use blocks."""
    return "".join(b.text for b in response.content if b.type == "text").strip()


def run_spoke(spoke: str, prompt: str, verbose: bool = True) -> str:
    """Run one spoke to completion in ISOLATION and return its report text.

    The isolation is structural, not a convention: `messages` is created empty here on
    every call. The spoke cannot see the coordinator's history, the other spoke's
    history, or its own history from a previous round. Anything it needs, the hub wrote
    into `prompt`. Anything the hub left out does not exist as far as this call is
    concerned - which is why a thin spoke report is almost always a hub bug.
    """
    system, tools = SPOKES[spoke]
    messages: list[dict] = [{"role": "user", "content": prompt}]
    transcript: list[str] = []  # text emitted along the way, kept in case the cap fires
    iteration = 0

    while True:
        iteration += 1
        if iteration > MAX_SPOKE_ITERATIONS:
            log.warning(
                "%s hit the %d-iteration cap - forcing a write-up", spoke,
                MAX_SPOKE_ITERATIONS,
            )
            # Don't just bail. A spoke that spends its whole budget on tools has done the
            # research and simply never written it down, and returning scraps makes the
            # next coverage check read as a gap it never addressed - so the hub
            # re-delegates work that was actually done, which is how a refinement loop
            # stalls at a coverage number that will not move. One more turn with no tools
            # converts what it gathered into the report it owed.
            instruction = (
                "STOP searching - you have reached your tool budget. Write your final "
                "report now, from what you have already gathered, covering every assigned "
                "subtopic. Write 'NO SOURCES FOUND' under any subtopic you never got to."
            )
            last = messages[-1]
            if last["role"] == "user" and isinstance(last["content"], list):
                # Don't start a second consecutive user message - append to this one.
                last["content"].append({"type": "text", "text": instruction})
            else:
                messages.append({"role": "user", "content": instruction})

            final = client.messages.create(
                model=MODEL, max_tokens=MAX_TOKENS, system=system, messages=messages
            )
            return text_of(final) or "\n\n".join(t for t in transcript if t)

        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            tools=tools,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})
        transcript.append(text_of(response))

        if response.stop_reason == "tool_use":
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                content, is_error = dispatch(block.name, block.input)
                if verbose:
                    arg = json.dumps(block.input)[:90]
                    hit = "miss" if "No results" in content else f"{len(content)}ch"
                    print(f"      {spoke}: {block.name}({arg}) -> {hit}")
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": content,
                        "is_error": is_error,
                    }
                )
            messages.append({"role": "user", "content": results})
            continue

        if response.stop_reason not in ("end_turn", "stop_sequence"):
            log.warning("%s stopped on %s", spoke, response.stop_reason)

        return text_of(response)


# ---------------------------------------------------------------------------
# 3. The hub - decomposition, delegation, aggregation, refinement
# ---------------------------------------------------------------------------

COORDINATOR_SYSTEM = (
    "You are the coordinator in a hub-and-spoke research system. You do no research "
    "yourself. You own four things and they are the only things you own:\n"
    "  1. decomposing a broad topic into subtopics that span its full breadth,\n"
    "  2. deciding which subagent gets which assignment, and writing prompts complete "
    "enough to act on in isolation,\n"
    "  3. aggregating what comes back and judging honestly what is still missing,\n"
    "  4. re-delegating against those gaps until coverage is sufficient.\n\n"
    "Your subagents share no memory with you or with each other. They know only what you "
    "put in their prompt. A subagent that returns thin work was usually given a thin "
    "assignment - look at your own decomposition and your own prompt before you blame it.\n\n"
    "Breadth is your responsibility alone. A subagent assigned 'solar' cannot know that "
    "'geothermal' was in scope and will never tell you it is missing."
)


def structured_call(system: str, user: str, schema_tool: dict, max_tokens: int = 2048) -> dict:
    """One coordinator call whose answer is forced through a tool schema.

    tool_choice pins the response to `schema_tool`, so the result is a validated dict
    rather than prose to be regex-parsed. Every hub decision that another function has
    to branch on goes through here.
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        tools=[schema_tool],
        tool_choice={"type": "tool", "name": schema_tool["name"]},
    )
    # A forced tool call that runs out of tokens mid-JSON still returns a tool_use block,
    # just with fields missing - which surfaces as a KeyError somewhere unrelated later.
    # Catch it here, where the cause is still obvious.
    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            f"{schema_tool['name']} was truncated at max_tokens={max_tokens}; its arguments "
            "are incomplete. Raise max_tokens or ask it to produce less."
        )

    for block in response.content:
        if block.type == "tool_use":
            missing = set(schema_tool["input_schema"].get("required", [])) - set(block.input)
            if missing:
                raise RuntimeError(
                    f"{schema_tool['name']} omitted required field(s): {sorted(missing)}"
                )
            return block.input

    raise RuntimeError(f"forced tool call returned no tool_use (stop: {response.stop_reason})")


SUBMIT_DECOMPOSITION = {
    "name": "submit_decomposition",
    "description": "Submit the list of subtopics the topic decomposes into.",
    "input_schema": {
        "type": "object",
        "properties": {
            "subtopics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Short subtopic name, 1-4 words.",
                        },
                        "scope": {
                            "type": "string",
                            "description": "One sentence: what belongs under this subtopic.",
                        },
                    },
                    "required": ["name", "scope"],
                },
            }
        },
        "required": ["subtopics"],
    },
}


@dataclass
class Subtopic:
    name: str
    scope: str


@dataclass
class ContextPacket:
    """Everything a spoke will know. The single channel into an isolated process.

    It is a dataclass and not an f-string on purpose. The failure this guards against is
    silent: drop the research goal from a prompt and nothing errors - you just get a
    subagent optimising for the wrong objective and a report you cannot explain. Named
    fields make an omission visible at the call site.
    """

    spoke: str
    research_goal: str
    assignment: list[Subtopic]
    why_assigned: str
    prior_findings: str = ""
    known_gaps: list[str] = field(default_factory=list)
    targeted_queries: list[str] = field(default_factory=list)
    round_number: int = 1

    def render(self) -> str:
        lines = [
            f"RESEARCH GOAL: {self.research_goal}",
            "",
            f"You are the {self.spoke}. This is round {self.round_number} of a coordinated "
            "research effort. You have no memory of earlier rounds, so everything you need "
            "is written below.",
            "",
            f"WHY YOU WERE ASSIGNED THIS: {self.why_assigned}",
            "",
            "YOUR ASSIGNED SUBTOPICS:",
        ]
        for i, st in enumerate(self.assignment, start=1):
            # Scope lines can run long after consolidation merges several items; trim so
            # the assignment stays readable and the packet stays bounded across rounds.
            scope = st.scope if len(st.scope) <= 240 else st.scope[:237] + "..."
            lines.append(f"  {i}. {st.name} - {scope}")

        if self.known_gaps:
            lines += [
                "",
                "GAPS THE COORDINATOR HAS ALREADY IDENTIFIED (these are why you were "
                "called again - a previous round came back thin or empty here):",
            ]
            lines += [f"  - {g}" for g in self.known_gaps]

        if self.targeted_queries:
            lines += [
                "",
                "SUGGESTED STARTING POINTS (the coordinator wrote these against the gaps "
                "above; broad category-name queries already failed, so start specific and "
                "reformulate further if these still return nothing):",
            ]
            lines += [f"  - {q}" for q in self.targeted_queries]

        if self.prior_findings:
            lines += [
                "",
                "WHAT OTHER AGENTS HAVE ALREADY ESTABLISHED (do not re-derive this; extend "
                "it, and flag anything your own sources contradict):",
                self.prior_findings,
            ]

        lines += [
            "",
            "Cover every assigned subtopic explicitly, one '## <subtopic>' section each, "
            "even the ones where your answer is that you found nothing. An omitted section "
            "is indistinguishable from a missing source to the coordinator.",
        ]
        return "\n".join(lines)


# Words too generic to identify a subtopic - "solar energy" and "wave energy" share
# "energy", so matching on it would call every subtopic preserved.
_GENERIC = {
    "energy", "power", "technology", "technologies", "system", "systems", "and", "the",
    "for", "with", "generation", "advanced", "emerging", "based", "conversion", "other",
    "resources", "sources", "production", "integration", "management", "scale",
}


def _distinctive(name: str) -> set[str]:
    """The tokens that actually identify a subtopic, generic filler removed."""
    tokens = {t.strip("(),-/").lower() for t in name.replace("-", " ").split()}
    return {t for t in tokens if len(t) > 3 and t not in _GENERIC}


def _is_dropped(original: Subtopic, merged: list[Subtopic]) -> bool:
    """True if nothing in `merged` mentions `original` by name or in a scope line."""
    keys = _distinctive(original.name)
    if not keys:
        return False  # nothing distinctive to look for - assume it survived
    blob = " ".join(f"{m.name} {m.scope}".lower() for m in merged)
    return not any(k in blob for k in keys)


def apply_scope_requirements(
    subtopics: list[Subtopic], must_cover: list[str], verbose: bool = True
) -> list[Subtopic]:
    """Force subjects the brief mandates into the decomposition, wherever it landed.

    Autonomous decomposition answers "what does this topic contain?", which is not always
    the same question as "what did the person asking require?". Renewable energy is the
    clean example: fusion is not renewable - its fuel is mined lithium and deuterium - so
    a correct decomposition omits it about half the time, and a brief that says fusion is
    in scope is not corrected by arguing taxonomy.

    This is deliberately a coordinator input and not a prompt tweak. A requirement stated
    here is auditable and always applied; the same requirement hinted at in a prompt is
    neither. Anything injected is announced, so the report's breadth is never mistaken for
    something the decomposition found on its own.
    """
    if not must_cover:
        return subtopics

    blob = " ".join(f"{st.name} {st.scope}".lower() for st in subtopics)
    missing = [m for m in must_cover if m.lower() not in blob]

    for m in missing:
        subtopics.append(
            Subtopic(
                name=m,
                scope=(
                    f"Mandated by the research brief. The decomposition did not surface "
                    f"{m} on its own - it may sit outside a strict reading of the topic - "
                    f"but the brief puts it in scope, so it is researched and reported "
                    f"like any other subtopic, with the definitional caveat stated."
                ),
            )
        )

    if verbose:
        covered = [m for m in must_cover if m not in missing]
        print(f"  scope check: {len(covered)}/{len(must_cover)} required subjects already present")
        if missing:
            print(f"  injected by brief -> {', '.join(missing)}")

    return subtopics


def decompose(topic: str, verbose: bool = True) -> list[Subtopic]:
    """Break `topic` into subtopics spanning its breadth. TWO passes, deliberately.

    Pass 1 is the obvious enumeration. Pass 2 shows the model its own answer and asks
    what is missing - the categories a knowledgeable critic would say were skipped.

    The second pass is the whole fix for narrow decomposition. A single pass reliably
    returns the prominent members of a category and stops: solar, wind, maybe hydro. It
    is not that the model does not know geothermal exists - it is that nothing ever asked
    it to audit its own list for omissions. So this asks, explicitly, and merges.
    """
    first = structured_call(
        COORDINATOR_SYSTEM,
        textwrap.dedent(f"""\
            Decompose this research topic into subtopics: "{topic}"

            Requirements:
            - At least {MIN_SUBTOPICS} subtopics, and enough to span the topic's full breadth.
            - Enumerate the whole category space, not the famous members of it. Include
              minor, regional, early-stage and unfashionable categories.
            - Subtopics must be distinct, each researchable on its own.
            - Name specific things. "Emerging technologies", "other approaches" and similar
              catch-alls are not subtopics - a subagent handed one has nothing to research.
              If you are reaching for an umbrella, name what is under it instead.

            Submit via submit_decomposition."""),
        SUBMIT_DECOMPOSITION,
    )
    subtopics = [Subtopic(s["name"], s["scope"]) for s in first["subtopics"]]
    if verbose:
        print(f"  pass 1 -> {len(subtopics)}: {', '.join(s.name for s in subtopics)}")

    # Pass 2: the breadth audit.
    listing = "\n".join(f"- {s.name}: {s.scope}" for s in subtopics)
    audit = structured_call(
        COORDINATOR_SYSTEM,
        textwrap.dedent(f"""\
            A coordinator decomposed the topic "{topic}" into these subtopics:

            {listing}

            You are auditing this list for BREADTH FAILURES before any work is delegated.
            A reviewer who knows this field well is about to read the finished report and
            ask "how did they not cover X?". Name every X.

            Look specifically for:
            - established members of this category omitted because they are less prominent
              than the ones already listed,
            - frontier, experimental or pre-commercial members of it,
            - THE DEFINITIONAL EDGE. Technologies a strict reading of the topic excludes
              but that are routinely discussed alongside it, and whose absence a reader
              would read as an oversight rather than as a definition being applied. Name
              them and put the caveat in the scope line ("outside the strict definition
              because X; included because Y"). Leaving them out is the more damaging
              choice: a caveated section costs a sentence, a conspicuous hole costs the
              report its credibility. Work through this bullet explicitly before you
              submit - it is the one most often skipped.
            - enabling or cross-cutting dimensions everything listed depends on.

            Three rules on what you submit:

            0. MATCH THE GRANULARITY OF THE LIST. Add missing peers of the items already
               there - never narrower slices of something already present in any form.
               Splitting one listed item into four sub-variants adds no breadth, and it
               costs real breadth later: the list then has to be consolidated back down,
               and that merge is what collapses genuinely distinct technologies into
               joint subtopics no one can assess.


            1. NAME SPECIFIC THINGS. "Emerging technologies", "other approaches", "niche
               methods" are not subtopics - they are the absence of a decomposition. A
               subagent handed an umbrella like that has no idea what to research and will
               return whatever it happens to think of, which is the failure you are here to
               prevent. If the umbrella is hiding three real technologies, name the three.
            2. STAY INSIDE THE TOPIC. Add members of this category and its genuine
               frontier; do not annex whole neighbouring domains that a reader would expect
               to find in a different report.

            Submit ONLY the missing subtopics via submit_decomposition - do not repeat any
            already listed above. Submit an empty array only if the list is genuinely
            exhaustive."""),
        SUBMIT_DECOMPOSITION,
    )

    have = {s.name.lower() for s in subtopics}
    added = [
        Subtopic(s["name"], s["scope"])
        for s in audit.get("subtopics", [])
        if s["name"].lower() not in have
    ]
    subtopics += added
    if verbose:
        print(
            f"  pass 2 (breadth audit) -> +{len(added)}: "
            f"{', '.join(s.name for s in added) or 'nothing missing'}"
        )

    # Consolidation. The audit deliberately over-generates, so merge back to a delegable
    # number - by combining related items, never by dropping any. Note the asymmetry: this
    # runs only when the list is too LONG. There is no equivalent rescue when it is too
    # short, because nothing downstream can recover a category the hub never named.
    if len(subtopics) > MAX_SUBTOPICS:
        listing = "\n".join(f"- {s.name}: {s.scope}" for s in subtopics)
        merged = structured_call(
            COORDINATOR_SYSTEM,
            textwrap.dedent(f"""\
                This decomposition of "{topic}" has {len(subtopics)} subtopics, more than
                the {MAX_SUBTOPICS} that can be delegated well:

                {listing}

                Consolidate to at most {MAX_SUBTOPICS} by MERGING related subtopics. Do not
                drop anything: every specific technology named above must still appear by
                name in some subtopic's scope line, so a subagent reading only the merged
                list still knows it is in scope.

                What to merge, in order of preference:
                - narrow variants into their parent technology,
                - enabling, cross-cutting, materials and policy items into fewer buckets -
                  merge these aggressively.

                Two hard constraints on the result:
                - No catch-all names. "Emerging technologies", "other", "miscellaneous" are
                  forbidden as merged subtopic names. Merging specific items into an
                  umbrella is not consolidation, it is deleting them: the name is what the
                  subagent researches and what the coverage check scores. If something has
                  no specific parent to merge into, leave it as its own subtopic and merge
                  elsewhere instead.
                - Only merge things that genuinely belong together. Two unrelated categories
                  sharing a subtopic because both were small is worse than exceeding the
                  cap - if that is the only way to hit the number, return one more subtopic
                  than asked for.

                What NOT to merge: two distinct primary technologies of this topic must not
                be combined into one subtopic. Coverage is scored per subtopic, so a merged
                pair lets a well-sourced half carry a thin half to a passing grade and the
                gap never gets found. Granularity here sets the resolution of every
                coverage check downstream.

                Submit via submit_decomposition."""),
            SUBMIT_DECOMPOSITION,
            max_tokens=3072,
        )
        candidate = [Subtopic(s["name"], s["scope"]) for s in merged["subtopics"]]
        if len(candidate) >= MIN_SUBTOPICS:
            # Verify the merge was lossless instead of trusting that it was. A merge that
            # quietly drops a category is indistinguishable, three steps later, from never
            # having thought of it - and by then nothing can recover it, because no spoke
            # is ever told to look and no coverage check is ever scored against it. So the
            # cap yields to breadth: anything that vanished comes back as its own subtopic.
            dropped = [st for st in subtopics if _is_dropped(st, candidate)]
            subtopics = candidate + dropped
            if verbose:
                print(
                    f"  consolidated -> {len(candidate)}: "
                    f"{', '.join(s.name for s in candidate)}"
                )
            if dropped:
                log.warning(
                    "consolidation dropped %d subtopic(s), restored: %s",
                    len(dropped), ", ".join(d.name for d in dropped),
                )
                if verbose:
                    print(f"  restored dropped -> {', '.join(d.name for d in dropped)}")
        else:
            log.warning("consolidation returned too few subtopics; keeping the long list")

    if len(subtopics) < MIN_SUBTOPICS:
        # Under-decomposition is the bug that produces a solar-and-wind report. Refuse to
        # proceed on it rather than delegate a narrow plan and debug the spokes later.
        raise RuntimeError(
            f"decomposition produced {len(subtopics)} subtopics, below the "
            f"MIN_SUBTOPICS={MIN_SUBTOPICS} floor - the hub has under-decomposed."
        )

    return subtopics


SUBMIT_COVERAGE = {
    "name": "submit_coverage",
    "description": "Submit a per-subtopic coverage assessment of the aggregated findings.",
    "input_schema": {
        "type": "object",
        "properties": {
            "assessments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "subtopic": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["covered", "partial", "missing"],
                            "description": (
                                "covered: specific sourced substance, enough to write a "
                                "real section. partial: mentioned but thin, generic or "
                                "without figures. missing: absent, or reported as "
                                "unsourced/not-in-corpus by both spokes."
                            ),
                        },
                        "evidence": {
                            "type": "string",
                            "description": (
                                "Concrete detail found, or why it falls short. At most two "
                                "sentences - this is a grading note, not a summary, and a "
                                "long one crowds out the subtopics still to be assessed."
                            ),
                        },
                        "gap": {
                            "type": "string",
                            "description": (
                                "What specifically is still needed, in one short sentence. "
                                "Empty if covered."
                            ),
                        },
                    },
                    "required": ["subtopic", "status", "evidence", "gap"],
                },
            }
        },
        "required": ["assessments"],
    },
}

SUBMIT_FOLLOWUPS = {
    "name": "submit_followups",
    "description": "Submit targeted follow-up assignments for the subtopics still short.",
    "input_schema": {
        "type": "object",
        "properties": {
            "search_queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Narrow queries for the web researcher, naming specific technologies, "
                    "metrics, named projects or milestones. Never a bare category name - "
                    "those already failed."
                ),
            },
            "document_questions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific questions to put to the document analyst.",
            },
        },
        "required": ["search_queries", "document_questions"],
    },
}


@dataclass
class Assessment:
    subtopic: str
    status: str
    evidence: str
    gap: str


def digest(findings: dict[str, str], limit: int = DIGEST_LIMIT) -> str:
    """Condense prior spoke output for forwarding into the next spoke's packet.

    Bounded because the packet is resent in full every round and would otherwise grow
    without limit. The truncation is announced rather than silent - a spoke told it is
    seeing an excerpt will not read the cut-off point as the end of what is known.
    """
    parts = []
    for spoke, text in findings.items():
        body = text if len(text) <= limit else text[:limit] + "\n[...excerpt truncated]"
        parts.append(f"### from {spoke}\n{body}")
    return "\n\n".join(parts)


# Ranking used to keep coverage monotonic across rounds.
_RANK = {"missing": 0, "partial": 1, "covered": 2}


def aggregate(
    topic: str,
    subtopics: list[Subtopic],
    findings: dict[str, str],
    retained: dict[str, Assessment] | None = None,
) -> list[Assessment]:
    """Combine both spokes' output and score coverage per subtopic.

    This is where the hub earns the name. A dispatcher would concatenate and stop; the
    judgement being made here - is this enough? - is what produces the gap list that the
    refinement loop feeds on.

    Coverage is monotonic, and that is load-bearing rather than an optimisation. An LLM
    grader re-scoring the same growing pile of evidence every round does not return the
    same verdict every round, so a subtopic judged covered in round 1 can come back
    partial in round 2 on wording alone. The loop then chases a number that moves in both
    directions and never converges - the first version of this code went 75% -> 60% while
    strictly gaining evidence. Since `findings` is append-only, no later round can make an
    earlier one's evidence insufficient, so a verdict is only ever allowed to improve, and
    settled subtopics are not re-graded at all.
    """
    retained = retained or {}
    # Only the unsettled subtopics go to the grader - cheaper, and a shorter list is
    # graded more consistently than a long one.
    open_subs = [
        st for st in subtopics
        if st.name not in retained or retained[st.name].status != "covered"
    ]
    if not open_subs:
        return [retained[st.name] for st in subtopics]

    combined = "\n\n".join(f"### {spoke} reported:\n{text}" for spoke, text in findings.items())
    listing = "\n".join(f"- {s.name}: {s.scope[:160]}" for s in open_subs)

    result = structured_call(
        COORDINATOR_SYSTEM,
        textwrap.dedent(f"""\
            Research goal: {topic}

            The subtopics that must be covered:
            {listing}

            Aggregated subagent findings:
            {combined}

            Assess each subtopic against the COMBINED body of findings above - the union
            of every source, across every round. You are asking one question only: is
            there enough here to write a substantive section on this subtopic?

            - "covered": the combined findings carry specific substance - figures, named
              projects, dates, costs, capacities. It does not matter which spoke supplied
              it, how many did, or whether the others found nothing.
            - "partial": named and discussed, but with no specific substance behind it.
            - "missing": nothing usable from any source.

            Where a subtopic names several technologies at once, grade the subtopic, not
            each technology inside it: "covered" means the findings support a substantive
            section on it, with the secondary members mentioned. Demanding full depth on
            every element of a joint subtopic marks it partial forever, no matter how much
            comes back.

            Three ways this assessment goes wrong, all of which re-delegate work that is
            already done and stall the loop:

            1. Grading against one source. The internal corpus being silent on a subtopic
               is NOT a gap if the web researcher supplied the figures - that is the two
               spokes working as intended, not a shortfall. "Not in corpus" is an answer
               about the corpus, never about the subtopic. Only call a subtopic missing
               when BOTH spokes came back empty on it.
            2. Grading against an ideal. A subtopic is not partial because a fuller
               treatment is imaginable. Perfection is not the bar; sufficiency is. Before
               writing "partial", name the specific fact that is missing - if you cannot,
               the subtopic is covered.
            3. Holding a verdict fixed. You are re-grading a body of evidence that has
               grown since it was last assessed. Read what is there now, not what you
               remember concluding.

            Use the subtopic NAME exactly as given above, with nothing appended - no scope
            text, no colon. Assess every subtopic, in order. Submit via submit_coverage."""),
        SUBMIT_COVERAGE,
        max_tokens=8192,
    )
    # The evaluator sometimes echoes the "name: scope" listing format back as the
    # subtopic. Snap each assessment onto the canonical name so that everything
    # downstream - gap lists, follow-up prompts, the report's section order - keys off
    # one spelling.
    canonical = {st.name.lower(): st.name for st in open_subs}
    fresh: dict[str, Assessment] = {}
    for a in result["assessments"]:
        raw = a["subtopic"].split(":")[0].strip()
        a["subtopic"] = canonical.get(raw.lower(), raw)
        fresh[a["subtopic"]] = Assessment(**a)

    merged: list[Assessment] = []
    for st in subtopics:
        prior = retained.get(st.name)
        new_a = fresh.get(st.name)
        if prior and new_a:
            # Keep whichever verdict is better - never let a re-grade walk one backwards.
            merged.append(new_a if _RANK.get(new_a.status, 0) > _RANK.get(prior.status, 0) else prior)
        elif new_a:
            merged.append(new_a)
        elif prior:
            merged.append(prior)
        else:
            merged.append(Assessment(st.name, "missing", "not assessed", "no findings"))
    return merged


def coverage_score(assessments: list[Assessment]) -> float:
    """Weighted completeness: covered=1.0, partial=0.5, missing=0.0."""
    if not assessments:
        return 0.0
    weights = {"covered": 1.0, "partial": 0.5, "missing": 0.0}
    return sum(weights.get(a.status, 0.0) for a in assessments) / len(assessments)


def plan_followups(topic: str, gaps: list[Assessment]) -> dict:
    """Turn a gap list into targeted queries - the hub's re-delegation decision.

    Re-sending the same assignment would produce the same empty result. The queries have
    to change, and making that a deliberate, structured step is what separates a
    refinement loop from a retry.
    """
    listing = "\n".join(f"- {g.subtopic} [{g.status}]: {g.gap or g.evidence}" for g in gaps)
    return structured_call(
        COORDINATOR_SYSTEM,
        textwrap.dedent(f"""\
            Research goal: {topic}

            These subtopics came back short:
            {listing}

            Write the follow-up assignments. Broad category-name queries have already been
            tried and failed, so do not repeat them - for each gap write queries naming
            specific technologies, named projects, installations, metrics, or milestones a
            published source would actually contain.

            One to three search queries per gap, plus a specific question per gap for the
            document analyst. Keep each one short. Submit via submit_followups."""),
        SUBMIT_FOLLOWUPS,
        max_tokens=3072,
    )


def render_coverage(assessments: list[Assessment]) -> str:
    """Coverage table - the diagnostic surface. A narrow decomposition is visible here."""
    mark = {"covered": "[x]", "partial": "[~]", "missing": "[ ]"}
    rows = [
        f"  {mark.get(a.status, '[?]')} {a.subtopic:<34} {a.status:<8} {a.evidence[:60]}"
        for a in assessments
    ]
    return "\n".join(rows)


def delegate_round(
    topic: str,
    subtopics: list[Subtopic],
    round_number: int,
    gaps: list[str] | None = None,
    queries: dict | None = None,
    prior: dict[str, str] | None = None,
    verbose: bool = True,
) -> dict[str, str]:
    """One full pass over both spokes. Sequential, and sequential on purpose.

    The document analyst runs second so its packet can carry the web researcher's
    findings. That ordering is the only mechanism by which one spoke's work can inform
    the other - they have no channel between them, so anything that crosses does so
    because the hub carried it.
    """
    prior = dict(prior or {})
    out: dict[str, str] = {}

    packets = {
        "web_researcher": (
            "You reach published external sources, which the internal corpus lacks. The "
            "coordinator needs figures nobody has written down internally."
        ),
        "document_analyst": (
            "You are the only agent that can see the organisation's internal corpus. The "
            "coordinator needs to know what these documents say - and, just as important, "
            "what they explicitly decline to cover."
        ),
    }

    for spoke, why in packets.items():
        packet = ContextPacket(
            spoke=spoke,
            research_goal=topic,
            assignment=subtopics,
            why_assigned=why,
            prior_findings=digest(prior) if prior else "",
            known_gaps=gaps or [],
            targeted_queries=(
                (queries or {}).get(
                    "search_queries" if spoke == "web_researcher" else "document_questions",
                    [],
                )
            ),
            round_number=round_number,
        )
        if verbose:
            print(f"    -> {spoke} ({len(packet.render())} chars of explicit context)")

        report = run_spoke(spoke, packet.render(), verbose=verbose)
        out[spoke] = report
        prior[spoke] = report  # so the next spoke this round sees it

    return out


def synthesise(topic: str, subtopics: list[Subtopic], findings: dict[str, str],
               assessments: list[Assessment], rounds: int) -> str:
    """Write the final report from the accumulated findings."""
    combined = "\n\n".join(f"### {s} reported:\n{t}" for s, t in findings.items())

    # Hand the coverage verdicts to the writer too. It is the difference between a thin
    # section that reads as though it were the whole story and one that says what is
    # missing - and the hub already knows which is which, so withholding it would make
    # the report less honest than the process that produced it.
    status = {a.subtopic: a for a in assessments}
    order_lines = []
    for i, st in enumerate(subtopics, start=1):
        a = status.get(st.name)
        note = ""
        if a and a.status != "covered":
            note = f"   [{a.status.upper()} - say so in the section: {a.gap or a.evidence}]"
        order_lines.append(f"{i}. {st.name}{note}")
    order = "\n".join(order_lines)

    response = client.messages.create(
        model=MODEL,
        max_tokens=8192,
        system=COORDINATOR_SYSTEM,
        messages=[{"role": "user", "content": textwrap.dedent(f"""\
            Write the final research report on: {topic}

            One '## ' section per subtopic, in this order, using exactly these names:
            {order}

            Accumulated findings from {rounds} round(s) of delegation:
            {combined}

            Rules:
            - Every subtopic gets a substantive section. Lead with concrete figures.
            - Use only what the subagents reported. Invent nothing. Where a subtopic is
              still thin, say so in its section rather than padding it out.
            - Note where the internal corpus and external sources disagreed, or where the
              corpus explicitly deferred a subject.
            - Open with a two-sentence scope statement and close with a '## Coverage note'
              recording which subtopics rest on a single source.
            - Search results were canned demo data; say so once in the scope statement."""
        )}],
    )
    return text_of(response)


def verify_required(report: str, must_cover: list[str]) -> dict[str, dict]:
    """Check the finished report against the brief, on the report's own text.

    Deliberately independent of the coverage assessment. That assessment grades the
    findings the spokes returned; this grades what actually reached the page. They come
    apart in the one direction that matters - evidence gathered but never written up -
    and only reading the report catches it.
    """
    lowered = report.lower()
    headings = [ln.lower() for ln in report.splitlines() if ln.lstrip().startswith("#")]

    out: dict[str, dict] = {}
    for term in must_cover:
        t = term.lower()
        mentions = lowered.count(t)
        in_heading = any(t in h for h in headings)
        out[term] = {
            "mentions": mentions,
            "in_heading": in_heading,
            # A section of its own, or repeated discussion, is substance. A single passing
            # mention in a list of things not covered is not.
            "substantive": in_heading or mentions >= 3,
        }
    return out


def research(topic: str, must_cover: list[str] | None = None, verbose: bool = True) -> dict:
    """The hub. Decompose -> delegate -> aggregate -> refine -> synthesise.

    Returns the report plus the audit trail: subtopics, per-round coverage, and the final
    assessment. The trail is the point - when a report comes back thin, you read the
    decomposition and the round-1 coverage table and you know immediately whether the hub
    never asked for the missing subject or asked and got nothing back.
    """
    print(f"\n=== COORDINATOR: {topic} ===\n")

    print("[1] decomposition")
    subtopics = decompose(topic, verbose=verbose)
    subtopics = apply_scope_requirements(subtopics, must_cover or [], verbose=verbose)
    print(f"  {len(subtopics)} subtopics (floor is {MIN_SUBTOPICS})\n")

    findings: dict[str, str] = {}
    history: list[float] = []
    assessments: list[Assessment] = []
    retained: dict[str, Assessment] = {}
    gaps: list[Assessment] = []
    queries: dict | None = None
    rounds = 0

    # The refinement loop. Exits on coverage, not on a round count - MAX_REFINEMENT_ROUNDS
    # is a cost backstop, and hitting it means reporting an honest shortfall rather than
    # pretending the threshold was met.
    while rounds < MAX_REFINEMENT_ROUNDS:
        rounds += 1
        label = "initial delegation" if rounds == 1 else f"refinement round {rounds - 1}"
        print(f"[2.{rounds}] {label}")

        assigned = subtopics if rounds == 1 else [
            Subtopic(g.subtopic, g.gap or g.evidence) for g in gaps
        ]
        new = delegate_round(
            topic,
            assigned,
            round_number=rounds,
            gaps=[f"{g.subtopic}: {g.gap or g.evidence}" for g in gaps],
            queries=queries,
            prior=findings,
            verbose=verbose,
        )
        for spoke, text in new.items():
            key = spoke if rounds == 1 else f"{spoke} (round {rounds})"
            findings[key] = text

        print(f"\n[3.{rounds}] aggregation and coverage evaluation")
        try:
            assessments = aggregate(topic, subtopics, findings, retained=retained)
            retained = {a.subtopic: a for a in assessments}
        except RuntimeError as exc:
            # Everything the spokes gathered is already in `findings`. Losing the whole
            # run because the grader for one round failed would throw away good research
            # over a bookkeeping step - stop refining and report what is in hand.
            log.warning("coverage evaluation failed (%s) - synthesising what we have", exc)
            assessments = list(retained.values())
            break
        score = coverage_score(assessments)
        history.append(score)
        print(render_coverage(assessments))
        print(f"  coverage: {score:.0%}\n")

        # "missing" before "partial": a subtopic with nothing behind it costs the report
        # a whole section, while a thin one still has something to print.
        gaps = sorted(
            (a for a in assessments if a.status != "covered"),
            key=lambda a: 0 if a.status == "missing" else 1,
        )
        if score >= COVERAGE_THRESHOLD or not gaps:
            print(f"  threshold met after {rounds} round(s) - no re-delegation needed\n")
            break

        if rounds >= MAX_REFINEMENT_ROUNDS:
            log.warning(
                "stopped at %.0f%% coverage after %d rounds with %d gap(s) open: %s",
                score * 100, rounds, len(gaps), ", ".join(g.subtopic for g in gaps),
            )
            break

        deferred = gaps[MAX_GAPS_PER_ROUND:]
        gaps = gaps[:MAX_GAPS_PER_ROUND]
        print(f"  {len(gaps)} gap(s) this round: {', '.join(g.subtopic for g in gaps)}")
        if deferred:
            print(f"  deferred to a later round: {', '.join(g.subtopic for g in deferred)}")
        print("  planning targeted follow-ups")
        try:
            queries = plan_followups(topic, gaps)
        except RuntimeError as exc:
            log.warning("follow-up planning failed (%s) - stopping refinement", exc)
            break
        for q in queries.get("search_queries", [])[:6]:
            print(f"    search: {q}")
        for q in queries.get("document_questions", [])[:6]:
            print(f"    docs:   {q}")
        print()

    print("[4] synthesis")
    report = synthesise(topic, subtopics, findings, assessments, rounds)
    print(f"  {len(report)} chars\n")

    required = verify_required(report, must_cover or [])
    if required:
        print("[5] brief verification - required subjects in the finished report")
        for term, info in required.items():
            tick = "[x]" if info["substantive"] else "[ ]"
            where = "own section" if info["in_heading"] else f"{info['mentions']} mentions"
            print(f"  {tick} {term:<14} {where}")
        met = sum(1 for i in required.values() if i["substantive"])
        print(f"  required-subject coverage: {met}/{len(required)}\n")

    return {
        "topic": topic,
        "subtopics": [s.name for s in subtopics],
        "must_cover": must_cover or [],
        "rounds": rounds,
        "coverage_history": history,
        "final_coverage": history[-1] if history else 0.0,
        "assessments": assessments,
        "required_coverage": required,
        "report": report,
    }


# ---------------------------------------------------------------------------
# 4. Entry point
# ---------------------------------------------------------------------------

DEFAULT_TOPIC = "renewable energy technologies"

# The brief that goes with DEFAULT_TOPIC. Five of these six turn up in the decomposition
# unprompted; fusion is the one that does not reliably, for the definitional reason in
# apply_scope_requirements(). Run with --no-brief to see the unaided decomposition.
DEFAULT_MUST_COVER = ["solar", "wind", "geothermal", "tidal", "biomass", "fusion"]


def main() -> int:
    args = sys.argv[1:]
    out_path = None
    if "--report" in args:
        i = args.index("--report")
        out_path = Path(args[i + 1])
        args = args[:i] + args[i + 2:]

    use_brief = "--no-brief" not in args
    args = [a for a in args if a != "--no-brief"]

    must_cover: list[str] = []
    if "--must-cover" in args:
        i = args.index("--must-cover")
        must_cover = [t.strip() for t in args[i + 1].split(",") if t.strip()]
        args = args[:i] + args[i + 2:]

    topic = " ".join(args) if args else DEFAULT_TOPIC
    if not must_cover and topic == DEFAULT_TOPIC and use_brief:
        must_cover = DEFAULT_MUST_COVER

    try:
        result = research(topic, must_cover=must_cover)
    except anthropic.APIError as exc:
        print(f"[api error: {exc}]")
        return 1

    print("=" * 78)
    print(result["report"])
    print("=" * 78)
    req = result["required_coverage"]
    met = sum(1 for i in req.values() if i["substantive"])
    print(
        f"\nrounds: {result['rounds']}  "
        f"coverage: {' -> '.join(f'{c:.0%}' for c in result['coverage_history'])}  "
        f"subtopics: {len(result['subtopics'])}"
        + (f"  required subjects: {met}/{len(req)}" if req else "")
    )
    if req and met < len(req):
        missing = [t for t, i in req.items() if not i["substantive"]]
        print(f"SHORTFALL - brief required but report does not cover: {', '.join(missing)}")

    if out_path:
        out_path.write_text(result["report"] + "\n")
        print(f"report written to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
