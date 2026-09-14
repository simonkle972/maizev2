"""A/B the contextualizer's off-topic check with and without the course summary.

    DOTENV_PATH=.env.local FLASK_SKIP_DOTENV=1 python eval/offtopic_classifier_test.py

Runs contextualize_query on every eval row twice, identical inputs, toggling only
Config.OFFTOPIC_COURSE_SUMMARY_ENABLED. No retrieval, no generation.
  catch rate       = bucket L rows classified off_topic      (want high)
  false deflection = every other row classified off_topic    (want zero)
session_context is None in both arms (no cached document), so a follow-up turn sees
its history but not a cached filename -- identical for both arms.
"""
import json, sys, collections, warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import app
from config import Config
from src.retriever import contextualize_query, get_course_summary

rows = [json.loads(l) for l in (ROOT / "eval" / "maize_eval_v1.jsonl").read_text().splitlines() if l.strip()]

def classify(row):
    r = contextualize_query(row["query"], row.get("prior_turns") or [], None, row["ta_id"])
    return r["intent"], r["reason"], r["fallback"]

out = {}
with app.app_context():
    Config.OFFTOPIC_COURSE_SUMMARY_ENABLED = True
    summaries = {ta: get_course_summary(ta) for ta in sorted({r["ta_id"] for r in rows})}
    for arm, flag in (("before", False), ("after", True)):
        Config.OFFTOPIC_COURSE_SUMMARY_ENABLED = flag
        with ThreadPoolExecutor(max_workers=8) as ex:
            out[arm] = dict(zip([r["row_id"] for r in rows], ex.map(classify, rows)))
        print(f"{arm}: done", flush=True)

res = []
for r in rows:
    b, a = out["before"][r["row_id"]], out["after"][r["row_id"]]
    res.append({"row_id": r["row_id"], "ta_id": r["ta_id"], "bucket": r.get("failure_type_target") or "working",
                "query": r["query"], "before": b[0], "after": a[0], "after_reason": a[1],
                "fallback": b[2] or a[2]})
(ROOT / "eval" / "offtopic_classifier_test_2026-09-13.json").write_text(
    json.dumps({"summaries": summaries, "rows": res}, indent=1, ensure_ascii=False))

L = [x for x in res if x["bucket"] == "L"]
O = [x for x in res if x["bucket"] != "L"]
print(f"\nclassifier fallbacks (call failures): {sum(x['fallback'] for x in res)}")
print(f"CATCH RATE (L, n={len(L)}):        before {sum(x['before']=='off_topic' for x in L)}/{len(L)}   after {sum(x['after']=='off_topic' for x in L)}/{len(L)}")
print(f"FALSE DEFLECTIONS (other, n={len(O)}): before {sum(x['before']=='off_topic' for x in O)}/{len(O)}   after {sum(x['after']=='off_topic' for x in O)}/{len(O)}")
print("  false deflections by bucket (after):", dict(collections.Counter(x["bucket"] for x in O if x["after"] == "off_topic")))
