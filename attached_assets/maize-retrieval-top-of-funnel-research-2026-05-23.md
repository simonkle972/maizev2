# Maize TA — Top-of-Funnel Retrieval Research

**Date:** 2026-05-23
**Scope:** Design question — should `contextualize_query()` keep its fixed 6-label intent taxonomy, or replace it with a richer LLM-driven decision layer?
**Audience:** Maize design decision, pre-implementation.

---

## Why this document exists

Today the top of Maize's retrieval funnel is a single gpt-4o-mini call that emits a JSON blob with four fields: `rewritten_query`, `intent` (one of 6 fixed labels), `current_focus`, `reason`. Downstream code dispatches on `intent`. The design critique on the table: "we've kind of presupposed six potential intents and then we have to shoehorn our way into those... we might be doing ourselves a disservice not to use [the model] to act on the very top of the funnel."

This report assembles what primary providers (Anthropic, OpenAI, Google) and serious practitioners actually say about that choice, then surfaces 3 design options with tradeoffs.

---

## Q1. Bucketed intent labels vs free-form structured action emission

### What the literature/practitioners say

The dominant pattern in 2024-2026 production systems is shifting **away from "LLM emits a label string, code dispatches"** and **toward "LLM emits a structured action (a tool call, a typed object) that already encodes both the decision and the parameters."** The two patterns are sometimes confused because both produce structured JSON, but their failure surfaces differ.

A label-based router is a classifier with side effects: the LLM picks one of N predefined strings, your code branches. Failures are familiar from pre-LLM NLP — misclassification under ambiguity, brittleness as new intents emerge, and "taxonomy drift" where the categories you wrote down a year ago no longer fit the queries users actually send. A 2025 Medium write-up by Armando Murga, surveying multi-domain agentic apps, notes that "as the number of intents increases (e.g., >50), even small errors can lead to significant downstream issues" and that "misclassified [intents] may result in subsequent agents executing logic on inaccurate premises" ([Murga 2025](https://medium.com/@mr.murga/enhancing-intent-classification-and-error-handling-in-agentic-llm-applications-df2917d0a3cc)). The arXiv "system-level failure taxonomy" paper lists "incorrect tool invocation" and "context-boundary degradation" among the 15 hidden failure modes in production LLM apps, of which mis-bucketed intent is a special case ([arXiv 2511.19933](https://arxiv.org/abs/2511.19933)).

A free-form structured-action router instead lets the model emit *the thing it would do* directly: a tool call with parameters, or a typed object containing route + arguments + reasoning. The classification is implicit. Misclassification still happens, but the parameters and reasoning are co-emitted, which means downstream code does not have to reconstruct intent from a label. Anthropic frames this distinction implicitly in their tool-use docs: "Claude decides when to call a tool based on the user's request and the tool's description" — there is no separate intent step ([Anthropic tool use overview](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)).

Maize's 6 labels (`continuation`, `concept_lookup`, `pivot`, `clarification`, `new`, `off_topic`) are exactly the kind of taxonomy that looks clean in a design doc but blurs in production. A real student message can be a `pivot` *and* a `clarification` at once ("wait, actually I meant problem 3 — what was the variance formula again?"). The current scheme forces the model to collapse that into one bucket.

### Concrete examples

- LangChain's "Routing in RAG Applications" piece spells out the LLM-completion router pattern (single-word label out, if/else branching) and explicitly acknowledges: "since a lot of the routing logic is based on using LLMs... which are non-deterministic in nature, we cannot guarantee that a router will always 100% make the right choice" ([Towards Data Science: Routing in RAG](https://towardsdatascience.com/routing-in-rag-driven-applications-a685460a7220/)). The same article shows semantic and zero-shot-classifier alternatives — all of which share the bucket problem.
- OpenAI's `orchestrating_agents` cookbook deliberately *avoids* the label pattern, using `transfer_to_X` tool calls instead: "A simple, but surprisingly effective way to do this is by giving them a `transfer_to_XXX` function." Routing is dispatched as a tool call ([OpenAI Cookbook: orchestrating_agents](https://developers.openai.com/cookbook/examples/orchestrating_agents)).

### Verdict — apply / refine / reject

**Refine.** Labels are not wrong, but they are an information-lossy intermediate representation. The richer pattern is: emit the action *and* the parameters in one structured object. Maize already emits `rewritten_query` alongside `intent`, so the architecture is half-way there.

---

## Q2. How Anthropic / OpenAI / Google handle this in their own products

### What the providers say

**Anthropic.** "Building Effective Agents" (Dec 2024, updated 2025) defines Routing as a workflow that "classifies an input and directs it to a specialized followup task" and explicitly says it "works well for complex tasks where there are distinct categories that are better handled separately, **and where classification can be handled accurately**" (emphasis added). It is conditional — Anthropic does not say "always classify first." Their consistent prior advice across the piece is to start simple: "finding the simplest solution possible, and only increasing complexity when needed. For many applications, optimizing single LLM calls with retrieval and in-context examples is usually enough" ([Anthropic — Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)).

Where Anthropic does emit structured choices, it leans on `tool_use` rather than label strings. Their "Advanced Tool Use" launch describes Claude orchestrating tools "through code rather than through individual API round-trips," and the strict-tool-use feature enforces schemas at the tool boundary ([Anthropic — Advanced Tool Use](https://www.anthropic.com/engineering/advanced-tool-use)). The signal: Anthropic's own primitive for top-of-funnel decisions is a typed tool call with parameters, not a free-text label.

**OpenAI.** The Cookbook's `orchestrating_agents` recipe is the cleanest primary expression of OpenAI's view. The triage agent emits a `transfer_to_X` tool call, and a single `Response` object carries both the active agent and the conversation state forward — multiple fields, one call ([OpenAI Cookbook: orchestrating_agents](https://developers.openai.com/cookbook/examples/orchestrating_agents)). The `Handling Function Calls with Reasoning Models` recipe further emphasizes that reasoning models can chain multiple function calls in series within one turn, which is structurally incompatible with a "label-first, dispatch-second" architecture ([OpenAI Cookbook: reasoning_function_calls](https://developers.openai.com/cookbook/examples/reasoning_function_calls)).

**Google (Gemini).** Google's function-calling docs do not recommend a separate intent-classification step at all. They describe **compositional function calling** ("chain multiple function calls together to fulfill a complex request") and **parallel function calling**, both of which assume the model is the routing decision-maker ([Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling)). The doc states that "the model determines when to call specific functions and provides the necessary parameters" — i.e., decision and parameters are co-emitted.

### Concrete examples

- Anthropic Claude Code (the product) does not classify user turns into a taxonomy before acting; it emits tool calls (Read, Edit, Bash, etc.) directly. The "intent" is implicit in *which* tool it picks. This is a deliberate product choice by Anthropic and is the closest in-product analogue to Maize's situation.
- OpenAI's GPTs / Assistants API similarly does no separate intent classification — tool selection *is* the routing.

### Verdict — apply / refine / reject

**Apply.** The consensus across all three primary providers is consistent: the LLM should emit a structured action (typically a tool/function call) carrying both the decision and the parameters, not a label that downstream code dispatches on. Anthropic's "start simple" caveat applies — but Maize is already past "single LLM call with retrieval"; the question is which complexity to add, and the providers' answer is "structured action emission," not "label classification + dispatch."

---

## Q3. Single-call multi-action emission

### What the literature/practitioners say

The pattern Maize is considering — one LLM call emitting `{rewritten_query, action, filter_hints, cache_action, reasoning}` — is well-trodden in 2025 production systems, and not just in toy examples.

The strongest primary example is OpenAI's `Response` object in `orchestrating_agents`: a single LLM turn returns a Pydantic object carrying multiple correlated fields (active agent, message history, tool calls). The Cookbook explicitly bundles "related outputs together" rather than splitting them across calls ([OpenAI Cookbook: orchestrating_agents](https://developers.openai.com/cookbook/examples/orchestrating_agents)).

Anthropic's strict tool use enables the same pattern: a tool with multiple required parameters forces the model to emit them together, schema-validated, in one call ([Anthropic tool use overview](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)). The token cost is small — Anthropic's table shows ~346 system tokens added to enable tool_use on Claude Opus 4.7, regardless of how many fields the tool has.

Practitioner data points:
- The Alhena.ai write-up on multi-turn RAG describes **four parallel contextualizers** (chat-history, product-search, query-expansion, filter-extraction), each its own LLM call to a different model. They explicitly *did not* fuse them, because each layer has a different latency/quality budget and a different optimal model ([Alhena: Query Rewriting Before Retrieval](https://alhena.ai/blog/query-rewriting-before-retrieval-multi-turn-rag/)). This is a real counterpoint: fusion is not always the right answer when the sub-tasks have different model needs.
- Sajal Sharma's "Comprehensive Agentic RAG" series shows a single structured output combining "datasource, reasoning, and routing decision" in one call ([Sharma, Agentic RAG Part 3](https://sajalsharma.com/posts/comprehensive-agentic-rag/)).
- The 2024 CHIQ paper finds that a **two-step** decomposition (disambiguate history, *then* rewrite) beats single-call rewriting on conversational search benchmarks, because the model can focus each step ([arXiv 2406.05013](https://arxiv.org/abs/2406.05013)). This cuts against fusion.

### Concrete examples

Maize's current 3 steps:
1. `contextualize_query()` LLM call → rewritten query + intent label
2. `analyze_query()` regex on the rewritten query → filename/category hints
3. Topic-switch heuristic → comparing focus to prior focus

A fused tool call could look like:
```json
{
  "name": "route_student_turn",
  "input": {
    "rewritten_query": "...",
    "action": "retrieve | use_cache | redirect_off_topic | clarify",
    "filter_hints": {"filename": "lecture 4", "category": "lectures"},
    "cache_action": "preserve | invalidate",
    "reasoning": "..."
  }
}
```

This *can* be one call. Whether it *should* be depends on whether the three current steps share enough context that fusion saves work, or whether they have different optimal models / different reliability needs.

### Verdict — apply / refine / reject

**Refine.** Single-call fusion is feasible and widely used, but CHIQ's evidence and Alhena's production architecture suggest you don't fuse for fusion's sake — fuse when the sub-decisions are correlated enough that the model benefits from joint reasoning. For Maize: `rewritten_query` and `filter_hints` are highly correlated (both derive from understanding the question), and `cache_action` is correlated with `action`. `reasoning` is essentially free with structured output. This is a strong fusion candidate.

---

## Q4. Context-clue use beyond intent classification

### What the literature/practitioners say

The user's framing — "this might be even broader than intent. It's just kind of like the usage of context clues" — aligns with where the conversational-RAG research has moved. The signals top-tier systems exploit include:

- **Unresolved coreference.** The CORAL benchmark finds that **60% of follow-up messages have unresolved coreferences** ("what about that?", "the second one") ([arXiv 2410.23090 — CORAL](https://arxiv.org/abs/2410.23090)). Maize's `rewritten_query` field already handles this — but only as a single rewrite, with no separate signal about *whether* a pronoun was resolved or *what referent* was picked.
- **Accumulated user state.** Alhena's contextualizers track "accumulated user needs (preferences/constraints learned from earlier turns, like 'under $200', 'for a home office')" and *reset* them on topic change to prevent context leakage ([Alhena](https://alhena.ai/blog/query-rewriting-before-retrieval-multi-turn-rag/)). The Maize analogue: a student working on problem 2, then pivoting to problem 4, should not carry the variance assumption from problem 2 into the new retrieval. The current `pivot` label triggers cache invalidation but does not capture *what* changed.
- **Topic-shift detection as its own signal.** The "Multi-Granularity Prompts for Topic Shift Detection" paper treats topic shift as a first-class detection problem with explicit features (semantic distance, discourse markers, entity overlap) rather than as a side effect of a single classification ([arXiv 2305.14006](https://arxiv.org/abs/2305.14006)).
- **Hesitation, self-correction, mode-switching markers.** "Wait", "actually", "going back to", "I think" — these are strong discourse cues that production conversational systems rarely surface explicitly to the LLM. The pattern in modern systems is to *not* hand-engineer features for these but to *let the LLM see the raw conversation and reason over it*, then emit structured signals (e.g., `correction_detected: true`, `previous_target: "problem 2"`, `new_target: "problem 3"`).
- **Document/topic history within session.** Maize already passes cached doc context. The richer pattern is to also pass *recent retrieval IDs* so the model can decide "the student is still on lecture 4" without re-retrieving.

The meta-point: the right abstraction is not "what label fits this turn" but "what observations does the next stage of the pipeline need?" That's a structured-output question, not a classification question.

### Concrete examples

A richer top-of-funnel output for Maize could surface signals the current 4-field design hides:
```json
{
  "rewritten_query": "...",
  "action": "retrieve",
  "filter_hints": {...},
  "cache_action": "preserve",
  "signals": {
    "coreference_resolved": true,
    "topic_continuity": "same_problem",
    "user_correction": null,
    "confidence": 0.85
  },
  "reasoning": "..."
}
```

This is still one LLM call; the model is just allowed to report richer observations.

### Verdict — apply / refine / reject

**Apply.** This is the strongest research-backed reason to redesign. The literature converges on: hand-written heuristics for context clues underperform letting the LLM see the raw conversation and emit structured observations. Maize is currently doing some of this (rewritten_query) but discarding most of it.

---

## Q5. Cost / latency for richer top-of-funnel

### What the literature/practitioners say

The practitioner consensus is clear: **routing should run on a small fast model, not a reasoning model**. Anthropic explicitly recommends in "Building Effective Agents" routing "easy/common questions to smaller models like Claude Haiku 4.5 and hard/unusual questions to capable models like Claude Sonnet 4.5" ([Anthropic — Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)).

Benchmarks as of 2026:
- **Claude Haiku 4.5**: ~597ms time-to-first-token on medium prompts, p95 612ms ([Kunal Ganglani — LLM Latency Benchmarks 2026](https://www.kunalganglani.com/blog/llm-api-latency-benchmarks-2026)).
- **gpt-4o-mini / GPT-5.4-mini**: $0.75/$4.50 per M tokens, sub-second TTFT typical.
- **Gemini Flash**: roughly 30% faster than GPT-5.4-mini on raw throughput per dev.to comparisons.

Adding 3-4 fields to a structured output adds essentially zero latency — the marginal output tokens are tens, not thousands. The Anthropic tool-use system-prompt overhead is ~346 tokens flat ([Anthropic tool use overview](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)). Maize's current call is reported as ~100-200ms gpt-4o-mini; a fused call with 4-5 more fields would land in roughly the same envelope.

Practitioner rule of thumb from the Iternal.ai 2026 model comparison: "Budget models handle 70-80% of production AI workloads; classification, summarization, extraction, simple generation, routing, formatting — none of these tasks need a premium model" ([Iternal.ai LLM Benchmarks 2026](https://iternal.ai/llm-selection-guide)).

The "different model per stage" pattern (Alhena uses GPT-4.1-nano for coreference, Gemini 2.5 Flash for product search) is real but only pays off above ~thousands of queries per day where the latency-quality optimization compounds. For Maize's volume, single-model top-of-funnel is the right call.

### Verdict — apply / refine / reject

**Apply.** Stay on a mini-tier model (gpt-4o-mini or migrate to Claude Haiku 4.5 / Gemini Flash). Richer structured output does not meaningfully change latency at this volume. Reserve gpt-5.2 for response generation, not routing.

---

## Recommended design options for Maize

Three concrete options, ranked by ambition. Each is grounded in the research above.

### Option A — Minimum-change refinement: keep buckets, add structured params

Keep the 6-label `intent` field for backward compatibility but co-emit `filter_hints`, `cache_action`, and a small `signals` object in the same call. Downstream dispatch logic stays the same; only `analyze_query()` regex parsing is deleted (its work moves into the LLM call).

- **Pros:** Smallest diff. Removes the regex step. No retraining downstream code. Recoverable if it underperforms.
- **Cons:** Keeps the brittle 6-label taxonomy. Doesn't address the "shoehorn" critique. Still loses information when a turn is genuinely two intents at once.
- **Risk:** Low.

### Option B — Tool-call router (provider-consensus pattern)

Replace `contextualize_query()` with a single tool call. Define one Claude/OpenAI tool — `route_student_turn` — whose schema is `{rewritten_query, action, filter_hints, cache_action, signals, reasoning}`. Drop the 6-label taxonomy entirely; `action` is a 3-4 value enum tied to actual downstream behavior (`retrieve`, `use_cache`, `redirect_off_topic`, optionally `clarify`). The 3 current steps (contextualize + analyze + topic-switch) collapse into one call.

- **Pros:** Matches what Anthropic, OpenAI, Google do in their own products. Eliminates label/dispatch mismatch — the action *is* the dispatch. Cleaner code path. Strict schema validation via `strict: true` on tool definition ([Anthropic strict tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)). Multi-field emission is exactly what CHIQ's two-step decomposition lost on, but Maize has more correlation between sub-decisions than CHIQ did, so fusion is the better tradeoff here.
- **Cons:** Larger refactor. Need to re-tune prompt with eval set. Cache-invalidation logic moves from a label switch to an enum switch (small risk of off-by-one bugs). Loss of the human-readable `intent` field unless preserved for logging.
- **Risk:** Medium. The literature is consistent that this is the right direction, but the rollout needs an A/B comparison on real prod logs before flip.

### Option C — Decompose like CHIQ / Alhena: multiple specialized contextualizers

Run 2-3 parallel small-model calls, each specialized: one for coreference + rewrite (gpt-4o-mini), one for filter extraction (Gemini Flash for structured output strength), one for topic-shift detection (could be a non-LLM classifier on embeddings). Fan in to dispatch.

- **Pros:** Each step can use the optimal model and prompt. Best empirical results on CHIQ's benchmark. Resilient: one stage failing doesn't poison the rest.
- **Cons:** Adds latency (parallel helps but max-of-N still > 1-call). More moving parts. Significant infra work for Maize's current scale. Alhena explicitly notes this pattern pays off when you have very different sub-task profiles — Maize's sub-tasks are similar (all "understand this turn in context"), so the gains are smaller than for product search.
- **Risk:** Higher. Probably overkill for current Maize volume; revisit if Option B underperforms or volume grows 10×.

### What the literature does not tell us

- **No primary benchmark on educational-RAG top-of-funnel specifically.** Most cited work is product search, customer service, or open-domain QA. Educational dialogues have unusual features (problem references, formula references, hesitation when stuck) that may not behave like e-commerce coreference.
- **No published guidance on cache-invalidation as a routed action.** The 6-label → cache decision in Maize is a custom pattern. The literature talks about routing to *retrieval strategies*, not to *cache state transitions*. This means Option B's `cache_action` field is a Maize-specific design choice with no direct prior art to lean on.
- **No clear evidence on fusion-vs-decomposition for sub-second educational use.** CHIQ argues for decomposition on offline benchmarks. Alhena argues for decomposition in production. OpenAI/Anthropic patterns argue for fusion. The decisive variable seems to be sub-task heterogeneity, which is moderate for Maize — leaning fusion, but not unambiguously.
- **No guidance on graceful failure of structured-action routers.** What happens if the tool call emits malformed parameters? Strict mode helps but doesn't fully solve it. Worth designing a fallback path before rollout.

---

## Sources

- [Anthropic — Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
- [Anthropic — Advanced Tool Use](https://www.anthropic.com/engineering/advanced-tool-use)
- [Anthropic — Tool use with Claude (docs)](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)
- [Anthropic Cookbook — orchestrator_workers pattern](https://github.com/anthropics/anthropic-cookbook/blob/main/patterns/agents/orchestrator_workers.ipynb)
- [OpenAI Cookbook — Orchestrating Agents](https://developers.openai.com/cookbook/examples/orchestrating_agents)
- [OpenAI Cookbook — Handling Function Calls with Reasoning Models](https://developers.openai.com/cookbook/examples/reasoning_function_calls)
- [Google — Gemini function calling docs](https://ai.google.dev/gemini-api/docs/function-calling)
- [arXiv 2406.05013 — CHIQ: Contextual History Enhancement](https://arxiv.org/abs/2406.05013)
- [arXiv 2410.23090 — CORAL Multi-turn Conversational RAG Benchmark](https://arxiv.org/abs/2410.23090)
- [arXiv 2305.14006 — Multi-Granularity Prompts for Topic Shift Detection](https://arxiv.org/pdf/2305.14006)
- [arXiv 2504.01018 — Self-Routing RAG](https://arxiv.org/html/2504.01018v1)
- [arXiv 2511.19933 — Failure Modes in LLM Systems: System-Level Taxonomy](https://arxiv.org/abs/2511.19933)
- [Towards Data Science — Routing in RAG Driven Applications](https://towardsdatascience.com/routing-in-rag-driven-applications-a685460a7220/)
- [Alhena — Query Rewriting Before Retrieval, Multi-Turn RAG](https://alhena.ai/blog/query-rewriting-before-retrieval-multi-turn-rag/)
- [Sajal Sharma — Comprehensive Agentic RAG, Part 3](https://sajalsharma.com/posts/comprehensive-agentic-rag/)
- [Armando Murga — Intent Classification in Agentic LLM Apps](https://medium.com/@mr.murga/enhancing-intent-classification-and-error-handling-in-agentic-llm-applications-df2917d0a3cc)
- [Iternal.ai — LLM Benchmarks 2026](https://iternal.ai/llm-selection-guide)
- [Kunal Ganglani — LLM API Latency Benchmarks 2026](https://www.kunalganglani.com/blog/llm-api-latency-benchmarks-2026)
