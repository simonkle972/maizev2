"""Build eval/real_followups_v1.jsonl from the prod QA-log exports in attached_assets/.

Every turn k >= 2 of a real multi-turn session becomes a row: query = the student's real
turn k, prior_turns = the real turns 1..k-1 with the REAL answers the TA gave. Unlabelled
(no correct document): the set exists for --generate + judge_pairs, where the question is
whether a generation change makes the reply respond to the conversation. The synthetic
eval rows carry one-line stub TA turns (median 66 chars), so a change to what the
generator sees of the conversation cannot show up on them; real answers run ~600 chars.

Prod TA ids are mapped onto their local copies; courses with no local copy are skipped.
"""
import csv, glob, json, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TA_MAP = {  # prod id -> local id
    "bv1COF3YbWV28OKv": "EgZ14pvqEYzfQRTM",   # econ-s1117 -> econ-s1117-local
    "EgZ14pvqEYzfQRTM": "EgZ14pvqEYzfQRTM",
    "z_B4fFY6jD1mhy9K": "z_B4fFY6jD1mhy9K",   # ec112-local
    "WBNtFkfPGZaJVQIk": "WBNtFkfPGZaJVQIk",   # mgt410-local
    "WlaRg6aDFG0NfTmZ": "z9m3H6mdgpQ-j6sr",   # mgt423 -> mgt423-local
}
sessions = {}
for path in sorted(glob.glob(str(ROOT / "attached_assets" / "Maize QA Master*.csv"))):
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        if not rd.fieldnames or "session_id" not in rd.fieldnames or "answer" not in rd.fieldnames:
            continue
        ts_col = next((c for c in rd.fieldnames if "time" in c.lower()), None)
        for idx, r in enumerate(rd):
            ta = TA_MAP.get(r.get("ta_id") or "")
            if not ta or not r.get("session_id") or not (r.get("query") or "").strip() or not (r.get("answer") or "").strip():
                continue
            if (r.get("adversarial_short_circuit") or "").lower() in ("true", "1"):
                continue
            key = (ta, r["session_id"])
            # Order by timestamp when the export has one, else by file order (exports are
            # chronological); dedupe across overlapping exports by the query text.
            order = (r.get(ts_col) if ts_col else None) or f"{Path(path).name}:{idx:05d}"
            bucket = sessions.setdefault(key, {})
            if not any(v["query"] == r["query"] for v in bucket.values()):
                bucket[order] = r
rows = []
for (ta, sid), turns in sorted(sessions.items()):
    ordered = [turns[k] for k in sorted(turns)]
    if len(ordered) < 2:
        continue
    for k in range(1, len(ordered)):
        prior = []
        for t in ordered[:k]:
            prior.append({"role": "user", "content": t["query"]})
            prior.append({"role": "assistant", "content": t["answer"]})
        rows.append({
            "row_id": f"real_{ta[:6]}_{sid[:8]}_t{k+1}", "source": "prod_log", "ta_id": ta,
            "query": ordered[k]["query"], "prior_turns": prior,
            "correct_doc_ids": [], "hard_negative_doc_ids": [], "forbidden_doc_ids": [],
            "failure_type_target": None, "expected_action": "retrieve", "expected_intent": {},
            "not_in_corpus": False,
            "notes": f"real session {sid}, turn {k+1} of {len(ordered)}; real answer at turn k: {len(ordered[k]['answer'])} chars; last prior TA answer {len(ordered[k-1]['answer'])} chars",
            "reference_answer": ordered[k]["answer"],
        })
out = ROOT / "eval" / "real_followups_v1.jsonl"
out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
import statistics as st
print(f"{len(rows)} follow-up rows from {sum(1 for v in sessions.values() if len(v) >= 2)} sessions; per TA:",
      {t: sum(1 for r in rows if r['ta_id'] == t) for t in set(r['ta_id'] for r in rows)})
print("last prior TA answer chars p50:", st.median(len(r["prior_turns"][-1]["content"]) for r in rows),
      "| rows where it exceeds 300:", sum(1 for r in rows if len(r["prior_turns"][-1]["content"]) > 300))
