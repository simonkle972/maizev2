"""Paired answer judge: which of two generated answers better responds to the student?

Phase 3 (2026-09-16). Retrieval metrics cannot see a change to what the generator is shown,
so two `run_eval.py --generate` runs (per-row JSON next to --out) are compared answer by
answer. Each pair is judged by gpt-5.2 in BOTH orders -- August 2026: a 16-3 verdict became
6-2-4 ties when every pair was re-judged both ways -- and only order-robust verdicts count.

Run:
    DOTENV_PATH=.env.local python eval/judge_pairs.py A.json B.json --label-a old --label-b new
    [--row-id r1,r2] [--failure-type K,I] [--rows-with-reply] [--out judged.md]

Prints agreed wins for A, wins for B, ties, and disagreements (the two orders disagreed),
with the judge's one-line reason for each row, and generation latency per side.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

EVAL_FILE = Path(__file__).parent / "maize_eval_v1.jsonl"

JUDGE_PROMPT = """You are grading two replies from a course teaching assistant to the SAME student message, in the same conversation.

CONVERSATION SO FAR (oldest first):
{transcript}

STUDENT'S CURRENT MESSAGE:
"{query}"

REPLY 1:
{a}

REPLY 2:
{b}

Which reply better responds to what the student actually asked, given the conversation? Judge on: (1) does it address the student's actual message and the point in the conversation it refers to (e.g. the TA's last step or question), (2) is it accurate and consistent with the conversation, (3) does it help the student make progress without simply giving away answers. Ignore length and formatting.

Answer with JSON only: {{"winner": "1" | "2" | "tie", "reason": "<one line>"}}"""


def load_rows():
    return {json.loads(l)["row_id"]: json.loads(l) for l in EVAL_FILE.open() if l.strip()}


def transcript_of(row) -> str:
    parts = []
    for t in row.get("prior_turns") or []:
        parts.append(f"{'Student' if t.get('role') == 'user' else 'TA'}: {t.get('content', '')}")
    return "\n\n".join(parts) or "(none)"


def judge_once(client, model, transcript, query, a, b) -> tuple[str, str]:
    from src.retriever import _json_completion
    prompt = JUDGE_PROMPT.format(transcript=transcript, query=query, a=a or "(empty)", b=b or "(empty)")
    r = _json_completion(client, model, prompt, 200)
    out = json.loads(r.choices[0].message.content)
    w = str(out.get("winner", "tie")).strip().lower()
    return ("1" if w in ("1", "reply 1") else "2" if w in ("2", "reply 2") else "tie"), str(out.get("reason", ""))[:160]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
    ap.add_argument("--row-id", default=""); ap.add_argument("--failure-type", default="")
    ap.add_argument("--rows-with-reply", action="store_true", help="include the *_reply_* rows")
    ap.add_argument("--model", default=None, help="judge model (default: Config.LLM_MODEL)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from config import Config
    from src.retriever import get_openai_client
    model = args.model or Config.LLM_MODEL
    rows = load_rows()
    A = {r["row_id"]: r for r in json.load(open(args.a))}
    B = {r["row_id"]: r for r in json.load(open(args.b))}
    ids = sorted(set(A) & set(B))
    if args.row_id:
        want = {x.strip() for x in args.row_id.split(",") if x.strip()}; ids = [r for r in ids if r in want]
    if args.failure_type or args.rows_with_reply:
        fts = {x.strip() for x in args.failure_type.split(",") if x.strip()}
        ids = [r for r in ids if (rows.get(r, {}).get("failure_type_target") in fts) or (args.rows_with_reply and "_reply_" in r)]
    ids = [r for r in ids if A[r].get("answer") is not None and B[r].get("answer") is not None]
    if not ids:
        sys.exit("no rows with answers on both sides")

    client = get_openai_client()
    wins_a = wins_b = ties = disagree = 0
    lines = [f"# Paired answer judge — {args.label_a} vs {args.label_b} ({model}, both orders)", ""]
    lines.append(f"| row | verdict | reason (order 1) |"); lines.append("|---|---|---|")
    for i, rid in enumerate(ids, 1):
        row = rows.get(rid, {}); tr = transcript_of(row); q = row.get("query", "")
        v1, why1 = judge_once(client, model, tr, q, A[rid]["answer"], B[rid]["answer"])
        v2, why2 = judge_once(client, model, tr, q, B[rid]["answer"], A[rid]["answer"])
        # map order-2 verdict back to A/B
        v2m = {"1": "2", "2": "1", "tie": "tie"}[v2]
        if v1 == v2m:
            verdict = {"1": f"{args.label_a} wins", "2": f"{args.label_b} wins", "tie": "tie"}[v1]
            if v1 == "1": wins_a += 1
            elif v1 == "2": wins_b += 1
            else: ties += 1
        else:
            verdict = f"disagree ({v1} / {v2m})"; disagree += 1
        print(f"[{i:3}/{len(ids)}] {verdict:18} {rid}", flush=True)
        lines.append(f"| {rid} | {verdict} | {why1} |")
    def ms(d, key): 
        vals = [d[r].get(key) or 0 for r in ids]; vals.sort(); return vals[len(vals)//2] if vals else 0
    summary = (f"\n**{len(ids)} rows.** {args.label_a} wins {wins_a} · {args.label_b} wins {wins_b} · ties {ties} · "
               f"disagreements {disagree} (excluded from wins). Generation p50 ms: {args.label_a} {ms(A,'generation_ms')} · {args.label_b} {ms(B,'generation_ms')}.")
    lines.insert(2, summary); print(summary)
    if args.out:
        Path(args.out).write_text("\n".join(lines) + "\n"); print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
