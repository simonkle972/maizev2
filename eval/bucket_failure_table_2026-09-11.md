| Bucket | n | fail % | What it tests |
|---|---:|---:|---|
| **A** | 7 | **14%** | Sibling assignments that bucket identically — same number, same doc_type. |
| **B** | 13 | **62%** | Sibling DOCUMENTS confused — Extra Problems I vs II, exam Part 1 vs Part 2 as separate files. |
| **C** | 10 | **30%** | A top-1 that looks plausible but is on the wrong topic entirely. |
| **D** | 8 | **62%** | Student is working a problem and the retriever hands back the solutions document. |
| **E** | 22 | **45%** | Cache anchoring — an earlier turn's document sticks across turns that should have moved on. |
| **F1** | 0 | — | Student names a new CONCEPT mid-conversation; retrieval should pivot at chunk level. |
| **F2** | 1 | **100%** | Student names a new DOCUMENT mid-conversation; retrieval should pivot at doc level. |
| **G** | 0 | — | Intra-doc structure — wrong section or sub-part of the RIGHT file. |
| **H** | 0 | — | Query names 2+ documents that must BOTH be surfaced. |
| **I** | 0 | — | Student corrects a prior wrong retrieval — the correction must win. |
| **J** | 0 | — | Ambiguous concept-vs-problem intent driving different retrieval. |
| **K** | 0 | — | Turn refers to the assistant's last message, not course material — should not re-retrieve. |
| **L** | 6 | **100%** | Off-topic or adversarial — should be deflected, not answered from course material. |
| **M** | 2 | **100%** | Material genuinely absent — correct answer is to say so, not to pick a document. |
| **N** | 1 | **100%** | No routable content and no prior turn to resolve it — measures the upstream gate, not routing. |
| **working** | 41 | **0%** | Already works. Exists to catch regressions, not to be fixed. |
| **ALL** | **111** | **33%** | Every row in the set. |

diagnostic (A-N): 37/70 = 53% fail
working:          0/41 = 0% fail

working rows by source: {'prod_log': 20, 'synthetic_working_case': 21}
