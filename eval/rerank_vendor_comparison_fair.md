# Reranker self-agreement — gpt-5.2 vs itself

_2026-08-19T02:32:11Z — 67 queries, 2 passes_

Baseline for the Cohere swap. gpt-5.2 is nondeterministic, so this is the ceiling any replacement should be judged against: if Cohere agrees with gpt-5.2 about as much as gpt-5.2 agrees with itself, they are equivalent.

- queries attempted: **67**
- reranked in at least one pass: **58**
- comparable (reranked in every pass): **58**

| metric | value |
|---|---|
| RBO (top-weighted) | **0.393** |
| overlap@5 | 0.383 |
| overlap@8 | 0.425 |
| top-1 agreement | 0.414 |
| identical top-8 ordering | 0.086 |

## Least stable queries

| row | RBO | ov@8 | top-1 |
|---|---|---|---|
| econ_s1117_real_typeD_06 | 0.000 | 0.00 | 0 |
| econ_s1117_real_working_lab_01 | 0.000 | 0.00 | 0 |
| econ_s1117_real_working_practice_01 | 0.000 | 0.00 | 0 |
| econ_s1117_real_working_paste_02 | 0.000 | 0.00 | 0 |
| econ_s1117_syn_typeB_02 | 0.000 | 0.00 | 0 |
| econ_s1117_syn_typeE_01 | 0.000 | 0.00 | 0 |
| econ_s1117_syn_typeE_02 | 0.000 | 0.00 | 0 |
| econ_s1117_syn_working_continuation_01 | 0.000 | 0.00 | 0 |
| econ_s1117_syn_working_01 | 0.000 | 0.00 | 0 |
| econ_s1117_syn_working_02 | 0.000 | 0.00 | 0 |

_Agreement measures equivalence, not quality._
