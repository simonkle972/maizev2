# ECON S1117 — document inventory (derived from prod logs)

**TA ID:** `bv1COF3YbWV28OKv` · **Course:** Introduction to Data Analysis and Econometrics

This inventory is derived from the `sources` and `hybrid_doc_filename` columns of the two prod log CSVs ([ECON S1117 test](../attached_assets/Maize%20QA%20Master%20[PROD]%20-%20ECON%20S1117%20test.csv) + [ECON S1117 doc-switching](../attached_assets/Maize%20QA%20Master%20[PROD]%20-%20ECON%20S1117%20test%20-%20expl%20doc%20switching.csv)). 30 unique documents were referenced across 44 rows.

**Identifier convention:** the strings below match `Document.display_name or Document.original_filename` exactly as they appear in the chunk `file_name` column. Eval rows reference these strings verbatim. Some names contain commas (e.g. `econ117_pset03- 2025, solutions`) — preserve them; never normalize.

## Problem sets / homework

| `file_name` | Type | Notes |
|---|---|---|
| `econ117_pset02_2025B_with_table` | problem | Problem Set 2 — discrimination problems. |
| `pset2_solutions-1` | solution | Solutions to PS 2. **Forbidden** when student asks PS 2 for help. |
| `econ117_pset03- 2025, solutions` | solution | Solutions to PS 3 (single doc, comma in name). **Forbidden** when student asks PS 3 for help. |
| `econS117_summer2024B_pset1-1` | problem | PS 1 from a different term (summer 2024). |
| `extra problems I-1-1` | problem | Extra problems I. Hard-negative for any "extra problems II" query (Type B). |
| `extra problems I - solutions-1` | solution | Solutions to extra problems I. |

## Labs

| `file_name` | Type | Notes |
|---|---|---|
| `Lab1` | problem | Lab 1. |
| `Lab2` | problem | Lab 2. Hard-negative for any "PS 2" query (Type A — both bucket to `doc_type=homework, assignment_number=2`). |
| `Lab4-1` | problem | Lab 4. |

## Practice problems / worksheets

| `file_name` | Type | Notes |
|---|---|---|
| `week 1.1. probabilities practice problems-1` | problem | Week 1 probabilities practice. |
| `week 1 practice problems - Random Variable, Central Tendencies-2-1` | problem | Week 1 practice — random variables (single doc, comma in name). |
| `week 1 solutions - Probabilities-2-1` | solution | Solutions to week 1 probabilities practice. |
| `week1_Probabilities-1-1` | problem | Week 1 probabilities (separate from the "practice problems" doc above). |
| `Central Tendencies-2-1` | problem | Central tendencies. |

## Quizzes (and quiz solutions)

| `file_name` | Type | Notes |
|---|---|---|
| `Quiz 2 solutions` | solution | Quiz 2 solutions. Often retrieved incorrectly for "quiz 1" queries (Type C). |
| `Quiz 3 - solutions 2025` | solution | Quiz 3 solutions. |

*Note: no `Quiz 1` problem doc was ever surfaced in the prod logs — the failures involve current retriever returning `Quiz 2 solutions` for "quiz 1" queries. The actual Quiz 1 doc may exist under a different filename (e.g., "econometrics quiz 1" — but we never saw it retrieved). Flag for verification.*

## Lectures (interactive + pre-recorded)

| `file_name` | Type |
|---|---|
| `Interactive Lecture 3-1-1` | lecture |
| `Interactive Lecture 5-1` | lecture |
| `Interactive Lecture 7-1` | lecture |
| `Interactive Lecture 9 slides-1` | lecture |
| `Pre-recorded lecture 02-2-1` | lecture |
| `Pre-recorded lecture 03-1` | lecture |
| `Pre-recorded lecture 06-1` | lecture |
| `pre-recorded lecture 04 - 2025` | lecture |

## Exams (final, midterm)

| `file_name` | Type | Notes |
|---|---|---|
| `econ117-final-fall-2018-1` | problem | 2018 final. Lexically lookalike for any "quiz 1" query — surfaced as Type C failure. |
| `final-fall-2019-1` | problem | 2019 final. |
| `final-fall-2019_Solutions-1` | solution | 2019 final solutions. |
| `econ117_s2019_final_complete-1` | problem | 2019 final, complete version. |
| `econ117_summer2022_midterm_Ztable-1-1` | problem | 2022 midterm with Z-table. |

## Known gaps (referenced in queries but not seen retrieved)

- **`extra problems II`** — Type B failure: student queries about "extra problems II problem 14" but retriever always returns `extra problems I-1-1`. Either the doc exists with a different display_name, or it's not indexed. **Investigate before eval runs.**
- **A "Quiz 1" problem doc** (canonical name unknown — student references it as "quiz 1" / "econometrics quiz 1") — Type C failure: retriever always returns `Quiz 2 solutions` or `econ117-final-fall-2018-1`. **Investigate before eval runs.**

## Doc-role classification (proposed, for the refined architecture's `doc_role` field)

| Role | Members |
|---|---|
| `problem` | econ117_pset02_2025B_with_table, econS117_summer2024B_pset1-1, extra problems I-1-1, Lab1, Lab2, Lab4-1, week 1.1. probabilities practice problems-1, week 1 practice problems - Random Variable, Central Tendencies-2-1, week1_Probabilities-1-1, Central Tendencies-2-1, econ117-final-fall-2018-1, final-fall-2019-1, econ117_s2019_final_complete-1, econ117_summer2022_midterm_Ztable-1-1 |
| `solution` | pset2_solutions-1, econ117_pset03- 2025, solutions, extra problems I - solutions-1, week 1 solutions - Probabilities-2-1, Quiz 2 solutions, Quiz 3 - solutions 2025, final-fall-2019_Solutions-1 |
| `lecture` | All Interactive Lecture * and Pre-recorded lecture * docs |
| `syllabus`/`reference` | (none observed in logs) |
