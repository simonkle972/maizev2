"""Round-trip between eval/eval_set_v2_working.csv (the human editing surface)
and eval/maize_eval_v1.jsonl (what the harnesses actually read).

    python eval/sync_eval_set.py --dry-run     # show what would be ingested
    python eval/sync_eval_set.py               # ingest new rows, then rewrite the CSV

Two directions in one command:
  INGEST    rows in the CSV with no matching JSONL row are built into proper eval
            rows -- row_id generated, doc names resolved against the live corpus,
            prior_turns reconstructed from the prod exports by session.
  REGENERATE the CSV is rewritten from the JSONL afterwards, so every cleanup
            (row_ids, resolved filenames, expected_action) is visible next time
            the file is opened. This is why edits must be made to NEW rows only;
            existing rows are owned by the JSONL.

Matching is by row_id when present, else by (ta_id, query) -- so a row that was
already ingested without its row_id written back is recognised, not duplicated.
"""
from __future__ import annotations
import argparse, csv, glob, json, re, subprocess, sys
from pathlib import Path

csv.field_size_limit(1 << 30)
ROOT = Path(__file__).resolve().parent.parent
JSONL = ROOT / 'eval' / 'maize_eval_v1.jsonl'
WORKING = ROOT / 'eval' / 'eval_set_v2_working.csv'
# Rows deliberately removed from the eval. Without this, deleting a row from the
# JSONL does not stick: the row is still in the CSV, so the next sync re-ingests it.
EXCLUDED = ROOT / 'eval' / 'excluded_rows.tsv'
EXPORTS = sorted(glob.glob(str(ROOT / 'attached_assets' / 'Maize QA*.csv')))

TA_NAME = {
    'EgZ14pvqEYzfQRTM': 'econ-s1117 (local copy of prod bv1COF3YbWV28OKv)',
    'z_B4fFY6jD1mhy9K': 'EC 112 Econ',
    'WBNtFkfPGZaJVQIk': 'Competitor',
    'iDYis09JtNUkyEJJ': 'Analytics101 (professor-created)',
}
SLUG = {'EgZ14pvqEYzfQRTM': 'econ_s1117', 'z_B4fFY6jD1mhy9K': 'ec112',
        'WBNtFkfPGZaJVQIk': 'competitor', 'iDYis09JtNUkyEJJ': 'analytics101'}

# The local dev DB is a COPY of the prod corpus under a different id, so prod
# export rows carry the prod id. Verified 2026-09-11: bv1COF3YbWV28OKv's source
# documents match econ-s1117-local's 70 docs. Needed to find prior turns.
PROD_ALIAS = {
    'EgZ14pvqEYzfQRTM': {'EgZ14pvqEYzfQRTM', 'bv1COF3YbWV28OKv'},
}

COLS = ['keep', 'needs_attention', 'label_correct_docs', 'label_mode',
        'label_supporting_docs', 'label_hard_negatives', 'label_forbidden_docs',
        'label_expected_action', 'label_verdict', 'label_note', 'row_id', 'ta_name',
        'ta_id', 'n_prior_turns', 'turn', 'failure_type', 'source', 'query',
        'prior_turns_json']

norm = lambda s: re.sub(r'[^a-z0-9]', '', (s or '').lower())
NULL_LABELS = {'none', 'n/a', 'na', '-', 'any relevant ones', 'any relevant one'}


def corpus_index(ta: str) -> dict:
    out = subprocess.run(
        ['docker', 'exec', 'maize_postgres_dev', 'psql', '-U', 'maize_dev',
         '-d', 'maize_ta_dev', '-t', '-A', '-c',
         f"SELECT COALESCE(display_name, original_filename) FROM documents WHERE ta_id='{ta}';"],
        capture_output=True, text=True).stdout
    return {norm(l): l.strip() for l in out.splitlines() if l.strip()}


def resolve(raw: str, index: dict, where: str) -> list:
    """Whole-string first (filenames may contain commas), then comma-split.
    Also tolerates the storage filename form: <randomprefix>_L6_Name.pdf"""
    raw = (raw or '').strip()
    if not raw or raw.lower() in NULL_LABELS:
        return []
    if norm(raw) in index:
        cand = [raw]
    elif '|' in raw:
        cand = [p.strip() for p in raw.split('|') if p.strip()]
    else:
        # legacy: some hand-pasted rows used commas. Whole-string already failed above,
        # and filenames may CONTAIN commas, so this is the last resort.
        cand = [p.strip() for p in raw.split(',') if p.strip()]
    out, bad = [], []
    for c in cand:
        if norm(c) in index:
            out.append(index[norm(c)]); continue
        s = re.sub(r'\.(pdf|pptx?|docx?|txt|md|xlsx?)$', '', c, flags=re.I)
        s = re.sub(r'^[A-Za-z0-9_\-]{8,}?_(?=L\d)', '', s)
        if norm(s) in index:
            out.append(index[norm(s)]); continue
        hits = [v for k, v in index.items() if norm(s) and (norm(s) in k or k in norm(s))]
        out.append(hits[0]) if len(hits) == 1 else bad.append(c)
    if bad:
        print(f'  !! UNRESOLVED ({where}): {bad}', file=sys.stderr)
    return out


_EXPORT_CACHE = None
def export_rows() -> list:
    global _EXPORT_CACHE
    if _EXPORT_CACHE is None:
        rows = []
        for f in EXPORTS:
            try:
                rows += [r for r in csv.DictReader(open(f, encoding='utf8')) if r.get('session_id')]
            except Exception:
                pass
        _EXPORT_CACHE = rows
    return _EXPORT_CACHE


def qkey(q: str) -> str:
    """Whitespace-insensitive match key. The exports contain doubled spaces and
    literal newlines that the hand-pasted CSV does not reproduce exactly."""
    return re.sub(r'\s+', ' ', (q or '')).strip().lower()[:60]


def prior_turns_for(query: str, ta: str) -> list:
    """Find this query in the prod exports and replay everything earlier in its session."""
    key = qkey(query)
    ids = PROD_ALIAS.get(ta, {ta})
    match = [r for r in export_rows()
             if qkey(r.get('query')) == key and (r.get('ta_id') or '') in ids]
    if not match:
        return []
    sess = match[0]['session_id']
    conv = sorted([r for r in export_rows() if r['session_id'] == sess],
                  key=lambda r: r.get('timestamp') or '')
    hist = []
    for r in conv:
        if qkey(r.get('query')) == key:
            break
        hist += [{'role': 'user', 'content': (r.get('query') or '').strip()},
                 {'role': 'assistant', 'content': (r.get('answer') or '').strip()}]
    return hist


OFFTOPIC = re.compile(
    r'champions league|president of the united states|color is the sky|'
    r'forget all previous instructions|capital of france', re.I)


def classify(row: dict, correct: list) -> dict:
    """Derive failure_type / expected_action / intent from the labels + notes."""
    q = row['query']
    act = (row.get('label_expected_action') or '').strip()
    ft = (row.get('failure_type') or '').strip() or None
    note = f"{row.get('needs_attention','')} {row.get('label_note','')}".lower()
    raw = (row.get('label_correct_docs') or '').strip().lower()

    if not act:
        if OFFTOPIC.search(q) or 'adversarial' in note or 'irrelevant' in note:
            act, ft = 'redirect', 'L'
        elif not correct and ('not in corpus' in note or raw in NULL_LABELS) and raw:
            act, ft = 'acknowledge_gap', 'M'
        elif not correct:
            act, ft = 'redirect', 'L'
        else:
            act = 'retrieve'
    cp = 'concept' if re.search(r'\b(what is|explain|when to use|cover|continuous or discrete)\b', q, re.I) else 'problem'
    return {'expected_action': act, 'failure_type': ft, 'concept_or_problem': cp}


def build(row: dict, index: dict, counters: dict) -> dict:
    ta, q = row['ta_id'].strip(), row['query'].strip()
    correct = resolve(row.get('label_correct_docs'), index, q[:40])
    c = classify(row, correct)
    if c['expected_action'] != 'retrieve':
        correct = []

    kind = c['failure_type'] or 'working'
    counters[(ta, kind)] = counters.get((ta, kind), 0) + 1
    tag = f"type{c['failure_type']}" if c['failure_type'] else 'working'
    rid = f"{SLUG.get(ta, ta)}_{'syn' if row.get('source')=='synthetic' else 'real'}_{tag}_{counters[(ta,kind)]:02d}_v2"

    # Explicit prior turns win over session lookup -- synthetic rows have no prod
    # session to reconstruct from, so the CSV must be able to carry them directly.
    pt = []
    raw_pt = (row.get('prior_turns_json') or '').strip()
    if raw_pt:
        try:
            pt = json.loads(raw_pt) or []
        except json.JSONDecodeError:
            print(f'  !! {q[:40]!r}: prior_turns_json is not valid JSON', file=sys.stderr)
    if not pt:
        pt = prior_turns_for(q, ta)
    declared = (row.get('n_prior_turns') or '0').strip()
    if declared.isdigit() and int(declared) and not pt:
        print(f'  !! {rid}: n_prior_turns={declared} but no session history found', file=sys.stderr)

    note = (row.get('needs_attention') or '').strip()
    ln = (row.get('label_note') or '').strip()
    notes = ' | '.join(x for x in (note, ln) if x) or 'added from working CSV'

    out = {
        'row_id': rid, 'source': (row.get('source') or 'prod_log').strip(),
        'ta_id': ta, 'query': q, 'prior_turns': pt,
        'correct_doc_ids': correct,
        'hard_negative_doc_ids': resolve(row.get('label_hard_negatives'), index, q[:40]),
        'forbidden_doc_ids': resolve(row.get('label_forbidden_docs'), index, q[:40]),
        'failure_type_target': c['failure_type'],
        'expected_intent': {
            'is_solution_request': False,
            'concept_or_problem': c['concept_or_problem'],
            # bucket I is DEFINED by an explicit correction, so the flag follows the
            # bucket. Hardcoding False silently invalidated every I row on ingest.
            'document_corrected_from_prior_turn': c['failure_type'] == 'I',
        },
        'notes': notes,
    }
    mode = (row.get('label_mode') or '').strip()
    if mode in ('any', 'all'):
        out['correct_doc_mode'] = mode
    if c['expected_action'] != 'retrieve':
        out['expected_action'] = c['expected_action']
        # Must cover every action. A two-way off_topic/concept_lookup branch gave
        # every no_retrieval row the wrong intent_class and failed validation.
        out['expected_intent']['intent_class'] = {
            'redirect': 'off_topic',
            'no_retrieval': 'clarification',
            'acknowledge_gap': 'concept_lookup',
        }[c['expected_action']]
    # the user's "answer was good even though retrieval didn't directly serve it" flag
    if re.search(r'answer was (still )?good|worked without|piecing together|without either doc', notes, re.I):
        out['answer_good_without_direct_source'] = True
    sup = resolve(row.get('label_supporting_docs'), index, q[:40])
    if sup:
        out['supporting_doc_ids'] = sup
    return out


def jsonl_rows() -> list:
    return [json.loads(l) for l in JSONL.read_text(encoding='utf8').splitlines() if l.strip()]


def regenerate_csv(rows: list) -> None:
    out = []
    for r in rows:
        n = len(r.get('prior_turns') or [])
        out.append({
            'keep': 'y',
            'needs_attention': '',
            'label_correct_docs': '|'.join(r.get('correct_doc_ids') or []),
            'label_mode': r.get('correct_doc_mode', ''),
            'label_supporting_docs': '|'.join(r.get('supporting_doc_ids') or []),
            'label_hard_negatives': '|'.join(r.get('hard_negative_doc_ids') or []),
            'label_forbidden_docs': '|'.join(r.get('forbidden_doc_ids') or []),
            'prior_turns_json': (json.dumps(r['prior_turns'], ensure_ascii=False)
                                 if r.get('prior_turns') else ''),
            'label_expected_action': r.get('expected_action', ''),
            'label_verdict': '',
            'label_note': (r.get('notes') or '')[:200],
            'row_id': r['row_id'],
            'ta_name': TA_NAME.get(r['ta_id'], r['ta_id']),
            'ta_id': r['ta_id'],
            'n_prior_turns': n,
            'turn': f'T{n//2+1}',
            'failure_type': r.get('failure_type_target') or '',
            'source': r.get('source', ''),
            'query': r.get('query', ''),
        })
    with open(WORKING, 'w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(out)
    print(f'regenerated {WORKING.name} from {len(out)} JSONL rows')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    existing = jsonl_rows()
    by_id = {r['row_id'] for r in existing}
    by_qt = {(r['ta_id'], (r.get('query') or '').strip()[:80]) for r in existing}

    excluded = set()
    if EXCLUDED.exists():
        for line in EXCLUDED.read_text(encoding='utf8').splitlines():
            if line.strip() and not line.startswith('#'):
                ta, _, q = line.partition('\t')
                excluded.add((ta.strip(), qkey(q)))

    csv_rows = list(csv.DictReader(open(WORKING, encoding='utf8')))
    pending = []
    dropped = set()
    for r in csv_rows:
        rid = (r.get('row_id') or '').strip()
        key = (r['ta_id'].strip(), qkey(r['query']))
        # keep=n is checked FIRST and applies to ALREADY-INGESTED rows too. It used to
        # sit after the "already in the JSONL" guard, which meant that once a row was
        # ingested it could never be removed through the CSV again.
        if (r.get('keep') or 'y').strip().lower() == 'n':
            excluded.add(key)
            if rid and rid in by_id:
                dropped.add(rid)
            continue
        if rid and rid in by_id:
            continue
        if key in by_qt:
            continue    # ingested on an earlier pass, row_id just wasn't written back
        if key in excluded:
            continue
        pending.append(r)

    if dropped:
        existing = [x for x in existing if x['row_id'] not in dropped]
        JSONL.write_text('\n'.join(json.dumps(x, ensure_ascii=False) for x in existing) + '\n',
                         encoding='utf8')
        print(f'removed {len(dropped)} previously-ingested rows marked keep=n')

    if excluded:
        EXCLUDED.write_text(
            '# Rows deliberately excluded from the eval. ta_id<TAB>query\n' +
            ''.join(f'{ta}\t{q}\n' for ta, q in sorted(excluded)), encoding='utf8')

    print(f'JSONL {len(existing)} rows | CSV {len(csv_rows)} rows | '
          f'{len(pending)} to ingest | {len(excluded)} excluded')
    if not pending:
        if not a.dry_run:
            regenerate_csv(existing)
        return

    tas = {r['ta_id'].strip() for r in pending}
    index = {ta: corpus_index(ta) for ta in tas}
    counters = {}
    for r in existing:
        m = re.search(r'_(type[A-Z]\d?|working)_(\d+)_v2$', r['row_id'])
        if m:
            k = (r['ta_id'], r['failure_type_target'] or 'working')
            counters[k] = max(counters.get(k, 0), int(m.group(2)))

    built = [build(r, index[r['ta_id'].strip()], counters) for r in pending]
    for b in built:
        print(f"  + {b['row_id']:40s} act={b.get('expected_action','retrieve'):16s} "
              f"ft={str(b['failure_type_target']):5s} pt={len(b['prior_turns']):2d} "
              f"{'AGWDS ' if b.get('answer_good_without_direct_source') else ''}"
              f"correct={b['correct_doc_ids']}")

    if a.dry_run:
        print('\n--dry-run: nothing written.')
        return

    with open(JSONL, 'a', encoding='utf8') as f:
        for b in built:
            f.write(json.dumps(b, ensure_ascii=False) + '\n')
    print(f'appended {len(built)} rows to {JSONL.name}')
    regenerate_csv(jsonl_rows())


if __name__ == '__main__':
    main()
