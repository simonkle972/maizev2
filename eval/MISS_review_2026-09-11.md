# MISS review — 2026-09-11 — APPLIED

Verdicts reviewed by Simon and applied to `eval/maize_eval_v1.jsonl` on 2026-09-11.
This file is a RECORD of what was done, not an input form. `bucket now` reflects the
live value in the JSONL, so it stays truthful if you reopen it.

| verdict | row | bucket then | bucket now | action taken |
|---|---|---|---|---|
| bad | `analytics101_real_typeA_01_v2` | A | A | no change |
| ok | `econ_s1117_real_typeB_01` | B | B | solutions docs removed from forbidden_doc_ids (policy: retrieval may surface answer keys; the generator withholds by content). |
| ok | `econ_s1117_real_typeD_05` | C | C | solutions docs removed from forbidden_doc_ids (policy: retrieval may surface answer keys; the generator withholds by content). |
| ok | `econ_s1117_syn_typeC_02` | C | C | solutions docs removed from forbidden_doc_ids (policy: retrieval may surface answer keys; the generator withholds by content). |
| ok | `econ_s1117_syn_typeC_03` | C | C | solutions docs removed from forbidden_doc_ids (policy: retrieval may surface answer keys; the generator withholds by content). |
| bad | `econ_s1117_real_typeD_01` | D | D | solutions docs removed from forbidden_doc_ids (policy: retrieval may surface answer keys; the generator withholds by content). |
| bad | `econ_s1117_real_typeD_02` | D | D | solutions docs removed from forbidden_doc_ids (policy: retrieval may surface answer keys; the generator withholds by content). |
| bad | `econ_s1117_real_typeD_03` | D | D | solutions docs removed from forbidden_doc_ids (policy: retrieval may surface answer keys; the generator withholds by content). |
| bad | `econ_s1117_real_typeD_06` | D | D | solutions docs removed from forbidden_doc_ids (policy: retrieval may surface answer keys; the generator withholds by content). |
| bad - because it's actually an adversarial query | `econ_s1117_syn_typeD_02` | D | D | no change |
| ok | `econ_s1117_real_paste_01` | E | E | solutions docs removed from forbidden_doc_ids (policy: retrieval may surface answer keys; the generator withholds by content). |
| bad | `econ_s1117_real_typeE_05` | E | E | solutions docs removed from forbidden_doc_ids (policy: retrieval may surface answer keys; the generator withholds by content). |
| bad | `econ_s1117_real_typeE_06` | E | E | solutions docs removed from forbidden_doc_ids (policy: retrieval may surface answer keys; the generator withholds by content). |
| bad | `econ_s1117_real_typeE_07` | E | E | solutions docs removed from forbidden_doc_ids (policy: retrieval may surface answer keys; the generator withholds by content). |
| bad, also in isolation looks more like a type D failure | `econ_s1117_real_typeE_08` | E | E | solutions docs removed from forbidden_doc_ids (policy: retrieval may surface answer keys; the generator withholds by content). |
| bad | `econ_s1117_real_typeE_11` | E | E | no change |
| bad | `econ_s1117_real_typeE_12` | E | E | no change |
| bad | `econ_s1117_real_typeE_13` | E | E | no change |
| bad | `econ_s1117_real_typeE_16` | E | E | solutions docs removed from forbidden_doc_ids (policy: retrieval may surface answer keys; the generator withholds by content). |
| bad | `econ_s1117_real_typeE_17` | E | E | solutions docs removed from forbidden_doc_ids (policy: retrieval may surface answer keys; the generator withholds by content). |
| bad | `mgt410_local_real_typeE_exam_part2_3b_01` | E | **C** | classified E->C; retrieval demonstrably returned a document without the asked-for content. |
| bad | `ec112_local_real_typeF2_homework_after_circular_flow_01` | F2 | F2 | no change |
| bad, also here it pull a wron doc not a wrong part since the exam solutions are actually divided into part 1 and 2 in terms of their files | `mgt410_local_real_typeG1_exam_part1_2a_01` | G1 | **B** | classified G1->B; retrieval demonstrably returned a document without the asked-for content. |
| bad, also here it pull a wron doc not a wrong part since the exam solutions are actually divided into part 1 and 2 in terms of their files | `mgt410_local_real_typeG2_exam_part1_2b_intradoc_sibling_01` | G2 | **B** | classified G2->B; retrieval demonstrably returned a document without the asked-for content. |
| bad, also here it pull a wron doc not a wrong part since the exam solutions are actually divided into part 1 and 2 in terms of their files | `mgt410_local_real_typeG2_exam_part1_2b_recovery_failed_01` | G2 | **B** | classified G2->B; retrieval demonstrably returned a document without the asked-for content. |
| ok | `econ_s1117_real_working_01_v2` | working | working | 'econS117_summer2024B_pset1-1' accepted as an alternative source (labeler verdict "ok"). |
| bad, it didn't pull the pset here - not sure why this was classsified as working | `econ_s1117_real_working_paste_02` | working | **N** | classified working->N. "I need help with this problem" with zero prior turns and nothing pasted — no routable content exists, so this measures the ups |
| badm it didn't pull the correct pset, not sure why classified as working | `econ_s1117_real_working_practice_03` | working | **E** | solutions docs removed from forbidden_doc_ids (policy: retrieval may surface answer keys; the generator withholds by content). |
| ok | `econ_s1117_real_working_quiz_01` | working | working | 'econometrics quiz 3' accepted as an alternative source (labeler verdict "ok"). |
| ok | `econ_s1117_real_working_quiz_02` | working | working | 'econometrics quiz 3' accepted as an alternative source (labeler verdict "ok"). |
| ok | `econ_s1117_syn_working_01` | working | working | solutions docs removed from forbidden_doc_ids (policy: retrieval may surface answer keys; the generator withholds by content). |
| ok | `econ_s1117_syn_working_02` | working | working | 'week 1 practice problems - Random Variable, Central Tendencies-2-1' accepted as an alternative source (labeler verdict "ok"). |
| ok | `mgt410_local_working_exam_part1_2a_BR_01` | working | **B** | classified working->B alongside the four identical rows; retrieval returned part_2_solutions, which contains none of the Part-1 content. |
| bad, it actually pull the wrong part of the exam which exists as two files for the solutions and as one without them, answer generation may have somehow been fine though given working status? | `mgt410_local_working_exam_part1_2a_eq_quantity_01` | working | **B** | classified working->B; retrieval demonstrably returned a document without the asked-for content. |
| bad, it actually pull the wrong part of the exam which exists as two files for the solutions and as one without them, answer generation may have somehow been fine though given working status? | `mgt410_local_working_exam_part1_2a_subproblem_01` | working | **B** | classified working->B; retrieval demonstrably returned a document without the asked-for content. |
| bad, it actually pull the wrong part of the exam which exists as two files for the solutions and as one without them, answer generation may have somehow been fine though given working status? | `mgt410_local_working_exam_part1_2a_total_Q_01` | working | **B** | classified working->B; retrieval demonstrably returned a document without the asked-for content. |
| bad, it actually pull the wrong part of the exam which exists as two files for the solutions and as one without them, answer generation may have somehow been fine though given working status? | `mgt410_local_working_exam_part1_2a_with_paste_01` | working | **B** | classified working->B; retrieval demonstrably returned a document without the asked-for content. |
| bad, it actually pull the wrong doc, answer generation may have somehow been fine though given working status? | `mgt410_local_working_exam_part2_3a_writeup_01` | working | **C** | classified working->C; retrieval demonstrably returned a document without the asked-for content. |
| bad, it actually pull the wrong doc, answer generation may have somehow been fine though given working status? | `mgt410_local_working_exam_part2_3b_explicit_recovery_01` | working | **C** | classified working->C; retrieval demonstrably returned a document without the asked-for content. |
