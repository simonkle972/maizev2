# Maize TA — Multi-turn Socratic Dialogue Readiness & Plan

**Created:** 2026-05-07 · **Status:** parked, awaiting ≥20-30 rows of prod data across multiple TAs/subjects before deciding which path to ship.

## Why this is parked

A small sample (~7 rows from one TA, MGT 425 macroeconomics) showed the model giving multi-step explanations of the answer mechanism in response to fresh T/F pastes — essentially handing over the cognitive work and asking the student to label T/F at the end. That's not Socratic. The user explicitly does not want a major `BASE_INSTRUCTIONS` change shipped from a tiny, single-subject sample. We need broader prod data first.

## Where the app stands today (structural readiness review)

**Foundations that already work:**

- **Conversation history**: last 10 messages pulled from `ChatMessage` per turn ([src/chat_streaming.py:248-255](../src/chat_streaming.py#L248)); 6 most recent fed to the LLM. Multimodal turns preserve images. Adequate.
- **Session cache** ([src/retriever.py:1735-1750](../src/retriever.py#L1735)): `session.active_context` persists `document_filename`, `document_content`, `problem_reference`, `attempt_counts`, `supplementary_content` across turns. Solution doc lazy-loaded after ≥2 student messages so we don't expose answers prematurely.
- **Intent classification** ([src/retriever.py:1417-1572](../src/retriever.py#L1417)): `contextualize_query()` distinguishes `continuation` / `clarification` / `concept_lookup` / `pivot` / `new` / `off_topic` and uses these to preserve or invalidate the cache. Contextualizer override aggressively keeps the cache when intent suggests we're still on the same problem.
- **Topic-switch detection**: structured heuristics handle "PS6 Q2 → PS6 Q5" (within-set switch) cleanly via `problem_reference` field comparison.
- **Patience escalation** ([src/response_generator.py:274-281](../src/response_generator.py#L274)): `get_patience_instructions(attempt_count)` tiers from "Try again!" → "one specific hint" → "walk through approach." Tied to `attempt_counts[problem_key]`.
- **Prompt cache architecture**: stable system message + dynamic system message split (commit `b52d259`) is exactly the structure where Socratic-mode rules belong — they'd go in the dynamic block.

**Gaps for sustained Socratic dialogue:**

| Gap | Severity | Where |
|---|---|---|
| No explicit "first turn = ONE Socratic question, do not lecture" rule | High | `BASE_INSTRUCTIONS` |
| `BASE_INSTRUCTIONS` SETUP rules optimize for quantitative; T/F / qualitative branch missing | High | `BASE_INSTRUCTIONS` lines 87-105 |
| No convergence signal — system can't tell "student is still working" vs "got it, wants write-up validation" | Medium | nowhere |
| Patience tiers tied to `attempt_count` are wrong-answer-shaped, not Socratic-pacing-shaped | Medium | `get_patience_instructions` |
| No "fading" semantics — scaffolding doesn't gradually reduce as student demonstrates competence | Medium | nowhere |
| Token budget tight at ~6-8 multimodal turns + full context | Low | `chat_streaming.py:252` (limit 10) |

## What 2026 best practice converges on

- **One question at a time on early turns** — Khanmigo's "reflective pauses instead of racing to the answer."
- **Adaptive depth** based on student confidence; ZPD scaffolding starts heavy, fades as competence demonstrates.
- **Convergence detection** is a first-class concern — recognize when the student has reached the answer; transition to "write it up" or "verify."
- **Question taxonomy**: question → questions about the question → questions about assumptions → questions about evidence → only then answer.
- **Platform logic, not just prompts**: track dialogue state explicitly so the model isn't relying on inferring "where are we in this problem" from history alone. Researchers consistently report that prompt-only Socratic mode drifts.
- **Differentiate question types**: T/F and conceptual need different scaffolding from formula-driven problems.

## Two paths to ship (decide after data)

### Path A — Prompt-only Socratic mode (~30 min)

- Add a "QUALITATIVE / T/F / CONCEPTUAL SETUP" section to `BASE_INSTRUCTIONS` parallel to the existing PROBLEM HELP: SETUP section.
- One Socratic opening question; no multi-step lecture; stop.
- Extend patience rules to cover Socratic pacing alongside wrong-answer escalation.
- **Pros:** cheap, reversible, one commit.
- **Cons:** prompt-only Socratic drifts over turns; convergence detection still absent; cross-subject generalization untested.

### Path B — Prompt rules + dialogue-state scaffolding (~half-day)

- Path A's prompt changes, plus:
- Extend `contextualize_query()` to add a `dialogue_state` field — values like `setup_needed`, `working_through`, `near_convergence`, `final_validation`. Inferred from conversation history + student's most recent message.
- Pass `dialogue_state` to the dynamic system message so the model gets explicit pacing guidance per turn ("working through — ask next Socratic question"; "near convergence — let them finish"; "final validation — confirm and offer write-up tweaks").
- **Pros:** explicit state tracking matches the Khanmigo/ZPD literature; far less drift; convergence handled cleanly.
- **Cons:** another LLM-classified field, slight latency (+50ms inside contextualizer budget), more moving parts.

## Decision criteria — what we need from the prod data

Once 20-30+ rows land across multiple TAs:

1. **Is the lecturing pattern present across subjects** (non-quant TAs too — language, history, philosophy)? If yes, Path B much more justified — prompt rules don't easily cover all subject types.
2. **Are conversations actually 5-8 turns long** in practice, or do students bail after 2-3? If short conversations dominate, Path A may suffice — drift becomes less of a concern when length is bounded.
3. **Do conversations show convergence-failure cases** (student says "got it" or gives a clean final answer and the model keeps questioning)? If yes, Path B's `dialogue_state` is genuinely needed.

## Sources informing this plan

- [Khanmigo Deep Dive — adaptive scaffolding & Socratic design](https://skywork.ai/skypage/en/Khanmigo-Deep-Dive:-How-Khan-Academy's-AI-is-Shaping-the-Future-of-Education/1972857707881885696)
- [AI Socratic Tutors: Teaching the World to Think](https://aicompetence.org/ai-socratic-tutors/)
- [SocraticLLM (CIKM 2024) — Socratic teaching for math](https://arxiv.org/abs/2303.08769)
- [Generative AI in Education: Socratic Playground for Learning (Princeton/2025)](https://arxiv.org/html/2501.06682v1)
- [A Theory of Adaptive Scaffolding for LLM-Based Pedagogical Agents (Aug 2025)](https://arxiv.org/html/2508.01503v1)
- [Dialogic Pedagogy for LLMs: Aligning AI with Proven Theories of Learning (June 2025)](https://arxiv.org/html/2506.19484v1)
- [The Socratic Prompt — Towards AI](https://towardsai.net/p/machine-learning/the-socratic-prompt-how-to-make-a-language-model-stop-guessing-and-start-thinking)
- [Socratic Prompting With AI — HALC AI Guide](https://halcaiguide.commons.gc.cuny.edu/ai-skills/socratic-prompting-with-ai/)
- [Khanmigo Review 2025 — AI Models Rank](https://www.aimodelsrank.com/reviews/khan-academy-khanmigo)
- [AI Tutors and Pedagogical Implementation — Khanmigo / Khan Academy](https://www.khanmigo.ai/)
