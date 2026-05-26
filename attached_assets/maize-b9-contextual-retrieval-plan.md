# B9 — Anthropic Contextual Retrieval (chunk prefix at index time)

**Status**: planned 2026-05-25. Indexing-only ship; batches with B8 (`b5287c9`) + B10 (`9eb81b5`) + eval expansion (`2a456a2`) for the next push event.

## Goal

Prepend a 1-2 sentence LLM-generated **chunk-specific context blob** to each chunk *before embedding*. Raw `chunk_text` stays for display; the embedded text gains self-description ("This chunk is from Problem Set 3, Question 2b, discussing Cournot equilibrium quantity competition…"). Per the published technique: 49% retrieval-failure reduction at one-time ingest cost.

## Why this is the right next step

1. **Per the audit's per-step verdict** ([maize-architecture-review-2026-05-23.md line 426](maize-architecture-review-2026-05-23.md)): "single highest-leverage indexing change in this audit. 49% retrieval-failure reduction at one-time ingest cost is the best precision/$ ratio in print."
2. **Indexing-first sequencing**: B8 + B10 are landed locally; their retrieval-side activations are gated on B9 anyway (a contextualised chunk gives the future `hybrid_doc_search` refactor + reranker simplification something useful to compare against). Doing B9 third on the indexing side closes the indexing trilogy before any retrieval-side cleanup begins.
3. **Unlocks downstream deletion**: B17 reranker simplification (gpt-5.2 MEDIUM → gpt-4o-mini or Cohere v3 once chunks self-describe), B19 supplementary teaching material removal (~140 lines), and finalizes B15 structural injection obsolescence.

## Scope

### 1. Schema — one new column on DocumentChunk

`DocumentChunk.contextual_prefix` (Text, nullable). Stores the generated 1-2 sentence prefix. Critical for observability — without it, debugging "why does this chunk match X" is impossible since the prefix is invisible inside the embedding. Also enables re-embedding without re-generating prefixes (cheaper rollback / iterate). Migration is one additive column; no data backfill of *old* chunks (they get the prefix populated as the re-index runs).

### 2. New function — `generate_contextual_prefix(full_doc_text, chunk_text, filename, content_title) -> str`

Location: `src/document_processor.py` (next to `summarize_doc`).

Single LLM call per chunk, with the full doc passed as static prefix. Uses **OpenAI `gpt-4o-mini` with automatic prompt caching** rather than adding the `anthropic` SDK:
- OpenAI prompt caching kicks in automatically for prefixes ≥1024 tokens, 50% discount on cached input. Since we process all chunks of one doc sequentially, the full-doc prefix gets cached for chunks 2+.
- Avoids adding a new SDK dep + a second API key path through config/.env/systemd. Matches the technique conceptually; the cost delta vs Claude+prompt-caching is small at our scale.
- Estimated cost: ~$0.20 for ECON S1117 (75 docs), ~$5-10 for full prod re-index — matches the audit's estimate.

Prompt (mirroring Anthropic's published example):
```
Here is the document, for context:
<document>{full_doc_text}</document>

Here is the chunk we want to situate within the whole document:
<chunk>{chunk_text}</chunk>

Please give a short succinct context (1-2 sentences) to situate this chunk
within the overall document for the purposes of improving search retrieval
of the chunk. Answer only with the succinct context and nothing else.
```

3 retries with exponential backoff; returns empty string on terminal failure (chunk still gets embedded with current section-enrichment behavior — graceful degradation).

### 3. Wire into `process_and_index_documents_resumable`

In the chunking loop (`src/document_processor.py` ~line 1574):
1. After `chunk_text_with_context` returns chunks, but BEFORE batched embedding,
2. For each chunk: `prefix = generate_contextual_prefix(full_doc_text, chunk["original_text"], ...)`
3. Build the embedding input as: `{prefix}\n\n[{filename} > {section}] {original_text}` (prefix prepended to existing `chunk_text_enriched`).
4. Persist `prefix` to new `contextual_prefix` column on the DocumentChunk row.
5. Batched embedding call uses the new prefixed input.

Idempotent guard: if `contextual_prefix` is already set on the chunk row, skip (matters for resumption + backfill).

### 4. Backfill script `scripts/backfill_chunk_contextual_prefix.py`

Mirrors `scripts/backfill_doc_summaries.py` pattern. Per-chunk operation (NOT per-doc re-index):
- For each Document (skip if all chunks already have `contextual_prefix`):
- Reconstruct full_doc_text from its chunks (`SELECT chunk_text FROM document_chunks WHERE document_id=X ORDER BY chunk_index`).
- For each chunk: generate prefix → re-embed `{prefix}\n\n{existing_enriched}` → persist both `contextual_prefix` + new `embedding` in one transaction.
- Args: `--ta-id <id>`, `--dry-run`, `--force`. Commits every 50 chunks.
- **Critical**: this approach lets us run B9 backfill without re-extracting docs (no PDF parsing, no vision calls). Reversible — if we want to revert, run a `--restore-no-prefix` flavor.

### 5. NO retrieval-side change

`retrieve_context`, `hybrid_doc_search`, reranker, supplementary teaching material — all untouched. The new prefix lives inside the embedding only.

## What ships in this session

Everything above as **one commit, local only**, batched with the three already-queued commits (`b5287c9` B8, `9eb81b5` B10, `2a456a2` eval) for the next push event.

## What's NOT in scope

- Prod re-index. That's a separate maintenance-window operation after local validation passes.
- B17 reranker simplification (gpt-5.2 → gpt-4o-mini / Cohere v3). Deferred — B9 unlocks it; the actual swap is a follow-up.
- B19 supplementary teaching material deletion (~140 lines in retriever.py). Deferred — B9 unlocks it; deletion is a follow-up after we see B17 still works without it.
- hybrid_doc_search refactor. Still gated on D12 multi-TA data + A/B testing.

## Verification

1. **Local smoke** — run backfill on ECON S1117 (~75 docs, est. ~500-1000 chunks). Spot-check 5 prefixes by reading them: do they accurately situate the chunk (doc + section + topic)? Verify embedding count unchanged, just regenerated.
2. **Indexing pipeline smoke** — upload one new doc to local ECON, trigger reindex of just that doc, confirm `contextual_prefix` populated on every chunk row + embedding shape unchanged.
3. **Eval validation gate** — re-run `eval/run_eval.py --ta-id EgZ14pvqEYzfQRTM` (warm cache, default). **Expectation: hit@5 holds or improves; F1/F2/G1/G2 buckets show improvement** (these are the failure types most likely to respond — chunk-routing now has self-description; doc-switching has more contextual disambiguation).
4. **No regression on working_case rows** — non-negotiable. If any working-case row flips MISS → HIT or vice versa, investigate the prefix's behaviour on that doc.
5. **Cross-TA re-run** — re-run eval on EC 112 + MGT 410 local (`--ta-id z_B4fFY6jD1mhy9K` + `--ta-id WBNtFkfPGZaJVQIk`) after local backfilling those corpora too. Validates the 49% lift isn't ECON-specific.

## Sequencing (concrete)

1. Migration: add `contextual_prefix` Text column on DocumentChunk. Hand-clean Alembic to strip the chat_message_images FK + bm25_tsvector GIN false-positives we always see.
2. `generate_contextual_prefix` function + smoke-test on a single doc.
3. Wire into `process_and_index_documents_resumable`.
4. Backfill script `scripts/backfill_chunk_contextual_prefix.py`.
5. Dry-run backfill on local ECON. Sanity-check chunk count + ~5 prefix outputs.
6. Real backfill on local ECON S1117 (~$0.20, ~20-30 min wall clock).
7. Spot-check ~10 prefixes: are they doc-section-topic-accurate?
8. Re-run 92-row eval (warm cache). Compare scorecard to today's.
9. Repeat (6)+(8) on EC 112 + MGT 410 local. Compare warm scorecards.
10. If all three hold/improve: commit, batch with prior three for next push.
11. Log B9 ship in `attached_assets/maize-architecture-review-2026-05-23.md` 3.3 + audit trail.

## Risks + rollback

- **Risk**: prefixes hallucinate or are too generic ("This chunk discusses economics"). Mitigation: spot-check after backfill; if quality is thin, tune prompt (one iteration is cheap at the per-chunk level since we have the column).
- **Risk**: embedding-space shift causes regression on some failure modes that current `[filename > section]` enrichment was carrying. Mitigation: the eval is the gate; if scorecard regresses we don't commit. If it ships and prod regresses, run reverse backfill (re-embed without prefix) — ~$5-10 to roll back. Reversible.
- **Risk**: full-doc context exceeds gpt-4o-mini's input limit (128k tokens) for huge docs. Mitigation: truncate to first ~80k chars (same approach as `summarize_doc`). Most academic docs are well under this.
- **Risk**: prompt-caching doesn't kick in as expected, costs balloon. Mitigation: cost-measure on the first 5 docs before committing to full corpus; OpenAI usage dashboard shows cached vs uncached tokens. If caching misses, we either accept the higher cost (~$1-2 for ECON) or revisit the SDK choice.

## Effort estimate

~3-4h total for the local-only ship (slightly more than B10 because of: full-doc prompt construction, per-chunk LLM call wiring in the hot indexing loop, more rigorous validation across 3 TAs).
