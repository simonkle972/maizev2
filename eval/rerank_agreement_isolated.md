# Reranker self-agreement — gpt-5.2 vs itself

_2026-08-18T02:48:12Z — 67 queries, 2 passes_

Baseline for the Cohere swap. gpt-5.2 is nondeterministic, so this is the ceiling any replacement should be judged against: if Cohere agrees with gpt-5.2 about as much as gpt-5.2 agrees with itself, they are equivalent.

- queries attempted: **67**
- reranked in at least one pass: **58**
- comparable (reranked in every pass): **58**

| metric | value |
|---|---|
| RBO (top-weighted) | **0.790** |
| overlap@5 | 0.814 |
| overlap@8 | 0.809 |
| top-1 agreement | 0.759 |
| identical top-8 ordering | 0.241 |

## Least stable queries

| row | RBO | ov@8 | top-1 |
|---|---|---|---|
| econ_s1117_real_working_lab_01 | 0.000 | 0.00 | 0 |
| econ_s1117_syn_working_continuation_01 | 0.000 | 0.00 | 0 |
| econ_s1117_syn_typeC_03 | 0.032 | 0.20 | 0 |
| econ_s1117_syn_typeA_03 | 0.330 | 0.75 | 0 |
| econ_s1117_real_typeA_01 | 0.497 | 0.20 | 1 |
| econ_s1117_real_typeA_03 | 0.497 | 0.20 | 1 |
| econ_s1117_real_typeD_02 | 0.497 | 1.00 | 1 |
| econ_s1117_syn_typeD_01 | 0.497 | 0.20 | 1 |
| econ_s1117_real_typeE_17 | 0.515 | 0.75 | 0 |
| econ_s1117_syn_typeD_03 | 0.569 | 0.62 | 0 |

_Agreement measures equivalence, not quality._
