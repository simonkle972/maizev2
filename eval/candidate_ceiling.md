# Candidate ceiling — can the correct document reach the reranker?

_2026-08-20T02:04:10Z — 49 labelled session openers, contextualizer off_

Four nested gates. Only the L1→L3 span is something a reranker can influence — everything lost at L1 is unrecoverable downstream, whichever vendor is chosen.

| gate | question | rate |
|---|---|---|
| L0 | label resolves to an indexed document | 45/49 = 92% |
| L1 | correct doc in the ≤5 document shortlist — **the ceiling** | 32/45 = **71%** |
| L2 | correct doc has ≥1 chunk in the ~20-chunk pool | 32/45 = **71%** |
| L3 | correct doc survives into the reranked top-8 | 30/45 = **67%** |

## Routing path — the split that matters

The direct-match short-circuit bypasses fusion and returns exactly ONE document, so on those queries the reranker chooses 8 chunks from a document that is already right or already wrong, and a wider shortlist cannot reach them at all.

| path | n | L1 | L2 | L3 |
|---|---|---|---|---|
| short-circuit fired | 22 | 86% | 86% | 86% |
| fusion ran | 23 | 57% | 57% | 48% |

- mean shortlist **2.9** docs; mean pool **16.8** chunks from **2.7** docs

## Funnel sweep

The `L1 fusion-only` column excludes short-circuit rows, which are inert to shortlist width by construction and otherwise dilute the effect.

| config | L1 | L2 | L3 | L1 fusion-only |
|---|---|---|---|---|
| docs=5,chunks=20 | 32/45 = 71% | 32/45 = 71% | 30/45 = 67% | 13/23 = 57% |
| docs=10,chunks=20 | 34/45 = 76% | 34/45 = 76% | 31/45 = 69% | 15/23 = 65% |
| docs=20,chunks=20 | 35/45 = 78% | 33/45 = 73% | 31/45 = 69% | 16/23 = 70% |
| docs=5,chunks=50 | 32/45 = 71% | 32/45 = 71% | 31/45 = 69% | 13/23 = 57% |

## L1 misses (13) — where the correct doc ranked

Ranks are 0-based within each fusion side's own pool. `-` means absent from that side, or the short-circuit returned before fusion ran.

| row | bm25 | dense | filename | short-circuit |
|---|---|---|---|---|
| econ_s1117_real_typeD_06 | - | - | - | YES |
| econ_s1117_real_typeE_17 | - | 17 | - |  |
| econ_s1117_real_working_quiz_01 | - | - | - | YES |
| econ_s1117_real_working_practice_03 | - | 4 | - |  |
| econ_s1117_real_working_paste_02 | - | 11 | - |  |
| econ_s1117_syn_typeA_03 | - | - | - |  |
| econ_s1117_syn_typeD_02 | - | - | - | YES |
| econ_s1117_syn_typeE_01 | - | 7 | - |  |
| econ_s1117_syn_working_continuation_01 | 8 | - | - |  |
| econ_s1117_syn_working_04 | - | - | - |  |
| mgt410_local_working_exam_part2_3a_01 | - | - | - |  |
| mgt410_local_working_exam_part2_3b_fresh_01 | - | - | - |  |
| mgt410_local_real_typeG1_exam_part1_2a_01 | - | - | - |  |

_Retrieval only: no reranker judgement affects L0–L2, which are fixed before `rerank()` is called._
