"""Import a TA exported from prod by eval/PULL_TA_FROM_VPS.md into the local dev DB.

Remaps documents.id / document_chunks.id so nothing collides with existing local rows,
and NULLs institution_id / professor_id (their FK targets don't exist locally and
retrieval never reads them).

    python eval/import_ta_dump.py scratchpad/ta_iDYis09JtNUkyEJJ.tar.gz [--dry-run]
"""
import argparse, csv, io, subprocess, sys, tarfile

csv.field_size_limit(1 << 30)
PSQL = ['docker', 'exec', '-i', 'maize_postgres_dev',
        'psql', '-U', 'maize_dev', '-d', 'maize_ta_dev', '-v', 'ON_ERROR_STOP=1']

def q(sql, quiet=False):
    r = subprocess.run(PSQL + ['-t', '-A', '-c', sql], capture_output=True, text=True)
    if r.returncode:
        print('SQL FAILED:', r.stderr.strip()[:600]); sys.exit(1)
    if not quiet and r.stdout.strip():
        print(r.stdout.strip())
    return r.stdout.strip()

def load(tar, name):
    f = tar.extractfile(name)
    return list(csv.DictReader(io.TextIOWrapper(f, encoding='utf8')))

def lit(v):
    if v is None or v == '':
        return 'NULL'
    return "'" + v.replace("'", "''") + "'"

def insert(table, cols, rows):
    """One multi-row INSERT, sent over stdin to avoid argv length limits."""
    vals = ',\n'.join('(' + ','.join(lit(r.get(c)) for c in cols) + ')' for r in rows)
    sql = f'INSERT INTO {table} ({",".join(cols)}) VALUES\n{vals};'
    r = subprocess.run(PSQL, input=sql, capture_output=True, text=True)
    if r.returncode:
        print(f'INSERT INTO {table} FAILED:', r.stderr.strip()[:800]); sys.exit(1)
    print(f'  inserted {len(rows)} into {table}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tarball')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    with tarfile.open(a.tarball) as t:
        ta = load(t, 'ta.csv')[0]
        docs = load(t, 'documents.csv')
        chunks = load(t, 'chunks.csv')

    ta_id = ta['id']
    print(f"TA {ta_id} '{ta['name']}' | {len(docs)} docs | {len(chunks)} chunks")

    if q(f"SELECT 1 FROM teaching_assistants WHERE id='{ta_id}';", quiet=True):
        print(f'TA {ta_id} already present locally. Delete it first (see runbook rollback).')
        sys.exit(1)

    # slug is UNIQUE across the table
    if q(f"SELECT 1 FROM teaching_assistants WHERE slug={lit(ta['slug'])};", quiet=True):
        ta['slug'] = (ta['slug'] + '-prodcopy')[:64]
        print(f"  slug collided -> {ta['slug']}")

    # FK targets don't exist locally; neither column is read by retrieval
    ta['institution_id'] = ''
    ta['professor_id'] = ''

    base_doc = int(q('SELECT COALESCE(max(id),0) FROM documents;', quiet=True))
    base_chunk = int(q('SELECT COALESCE(max(id),0) FROM document_chunks;', quiet=True))
    doc_map = {d['id']: base_doc + i + 1 for i, d in enumerate(docs)}
    print(f'  remapping doc ids from {base_doc+1}, chunk ids from {base_chunk+1}')

    for d in docs:
        d['id'] = str(doc_map[d['id']])
    for i, c in enumerate(chunks):
        c['id'] = str(base_chunk + i + 1)
        if c['document_id'] not in doc_map:
            print(f"  chunk {c['id']}: orphan document_id {c['document_id']}"); sys.exit(1)
        c['document_id'] = str(doc_map[c['document_id']])

    if a.dry_run:
        print('\n--dry-run: nothing written.')
        for d in docs:
            print(f"  doc {d['id']}: {d.get('display_name') or d['original_filename']}"
                  f"  [{d.get('doc_type') or '-'}/{d.get('doc_category') or '-'}]"
                  f"  full_text={len(d.get('full_text') or '')}b")
        return

    insert('teaching_assistants', list(ta.keys()), [ta])
    insert('documents', list(docs[0].keys()), docs)
    for i in range(0, len(chunks), 200):
        insert('document_chunks', list(chunks[0].keys()), chunks[i:i+200])

    q("SELECT setval('documents_id_seq', (SELECT max(id) FROM documents));", quiet=True)
    q("SELECT setval('document_chunks_id_seq', (SELECT max(id) FROM document_chunks));", quiet=True)

    print('\nVERIFY:')
    q(f"""SELECT t.id, t.name, t.is_indexed, count(DISTINCT d.id) AS docs,
                 (SELECT count(*) FROM document_chunks WHERE ta_id='{ta_id}') AS chunks
          FROM teaching_assistants t LEFT JOIN documents d ON d.ta_id=t.id
          WHERE t.id='{ta_id}' GROUP BY t.id, t.name, t.is_indexed;""")

if __name__ == '__main__':
    main()
