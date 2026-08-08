# Topic Graph: Syllabus-Extracted Topics + Auto-Tagging

**Status:** Planned, not started. Set aside on 2026-04-21 to revisit later.

## Context

Gap analysis against 6 research papers on AI tutoring systems identified four structural gaps in Maize: persistent learner models, misconception feedback loop, instructor recommendations, and empirical impact validation. Of those, this plan prioritizes **Gap 5 (course graph)** first because:
- Student persistence requires enrolled students; current pilots use admin-created public-link TAs with anonymous sessions
- Professor recommendations are incremental on existing analytics
- Misconception loop depends on persistent student state
- A topic graph produces immediate, felt improvements (retrieval bias, better analytics grouping) and unblocks later features (quizzes, objective-aware recommendations, coverage analysis)

## Approach

**Hierarchy:** Topics are extracted from the syllabus. Non-syllabus documents (lectures, homework, exams, readings) are auto-tagged to 1–N topics. `Document.doc_type` already handles the category axis — no new field needed there.

**Terminology:** We call them **topics** internally (e.g., "Savings", "ISLM", "Bayes theorem"), not "learning objectives." Smaller promise to the professor, matches what we can extract reliably.

**Extraction strategy:**
- Primary: LLM pass over the syllabus during/after indexing to extract topics
- Fallback: corpus-wide topic clustering (same prompt pattern as `cluster_topics()` in `src/analytics.py`) if syllabus yields fewer than 3 topics
- Both paths write to the same `Topic` table; professor sees a review screen to confirm/edit

**Tagging granularity:**
- Doc-level tagging (many-to-many), not chunk-level. Chunks inherit their parent doc's tags at retrieval time.
- Reason: simpler, less LLM cost, and sufficient for retrieval bias + analytics grouping in the first pass.

## Schema

New migration adds:

```python
class Topic(db.Model):
    id = Integer, primary_key
    ta_id = String(32), FK → teaching_assistants.id, indexed
    label = String(128)  # "Savings", "Bayes theorem"
    order = Integer  # for stable display
    source = String(32)  # 'syllabus' | 'corpus_fallback' | 'manual'
    created_at = DateTime

class DocumentTopic(db.Model):
    document_id = Integer, FK → documents.id
    topic_id = Integer, FK → topics.id
    confidence = Float  # LLM-reported 0-1
    PRIMARY KEY (document_id, topic_id)
```

## Critical Files to Modify

1. **`models.py`** — add `Topic` and `DocumentTopic` models + relationships
2. **`migrations/versions/<new>_add_topic_graph.py`** — new migration
3. **`src/document_processor.py`** — two new functions:
   - `extract_topics_from_syllabus(ta_id, syllabus_doc)` — LLM pass returning list of topic labels
   - `tag_document_to_topics(ta_id, document)` — LLM pass scoring document against existing topics
   - Hook into the indexing pipeline: after syllabus indexes, extract topics; after each non-syllabus doc indexes, tag to topics
4. **`src/retriever.py`** — add topic-bias in `retrieve_context()`:
   - Detect topic mentions in query (simple LIKE against `Topic.label`)
   - Boost chunks whose parent doc is tagged to matched topics (add ~0.1 to vector score, or ~2 to LLM rerank score)
5. **`src/analytics.py`** — update `get_top_challenges()` to group by topic when available, falling back to existing regex-based doc/problem pairing
6. **`professor.py`** — new routes:
   - `GET /professor/ta/<ta_id>/topics` — review/edit screen
   - `POST /professor/ta/<ta_id>/topics` — save edits (add/remove/rename)
7. **`templates/professor/manage_ta.html`** — link to the topics review page; show topic count on the TA card
8. **`templates/professor/topics.html`** (new) — review screen with topic list + per-topic tagged document count + reassign UI

## Existing Code to Reuse

- **`cluster_topics()` in `src/analytics.py`** — LLM clustering prompt shape is directly reusable for both syllabus extraction and corpus fallback
- **Indexing pipeline in `app.py`** (`run_indexing_task`) — new steps plug in after existing doc processing
- **`retrieve_context()` signature** in `src/retriever.py` — topic bias is an internal enhancement, no API change
- **`get_top_challenges()` in `src/analytics.py`** — add topic grouping as a superset

## Verification

1. **Syllabus extraction.** Upload a syllabus to a dev TA, run indexing. Check `Topic` table populated with 5–15 reasonable topic labels.
2. **Auto-tagging.** Upload a problem set to the same TA, run indexing. Check `DocumentTopic` rows created with plausible confidence scores.
3. **Corpus fallback.** Create a TA with only non-syllabus documents. Run indexing. Confirm fallback path triggers and topics extracted from corpus.
4. **Retrieval bias.** Ask a topic-specific question (e.g. "what is ISLM?") via test chat. Check QA log to confirm topic-tagged chunks were boosted and ranked higher than a pre-change baseline query.
5. **Analytics grouping.** Generate some chat traffic across several topics. Open analytics dashboard — confirm "Top Challenges" section groups by topic label.
6. **Review screen.** Load `/professor/ta/<id>/topics`. Confirm topic list + tagged docs visible, edit flow works, changes persist.

## Deferred Decisions

- **Quizzes-in-chat** intentionally scoped out of the first pass. Worth its own PRD once the topic graph is in place.
- **Student-facing progress view** blocked on persistent learner identity (Gap 1).
- **Objective coverage view** ("your materials cover Topics 1–7 well; Topic 8 is under-resourced") is a natural next step once tagging is live.
