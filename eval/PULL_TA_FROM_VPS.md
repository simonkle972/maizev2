# Runbook — copy TA `iDYis09JtNUkyEJJ` from prod into the local dev DB

**Read-only on prod.** Nothing below writes to, restarts, or reconfigures anything on the VPS.
Every step is a single command. Run them in order; stop if a CHECK doesn't match.

We export the *indexed* state (documents + chunks **with their embeddings**), not the source
files. That means no re-indexing and no OpenAI spend, and the local copy is byte-identical to
what prod actually retrieved against — which is the point, since we're reproducing prod failures.
`file_content` (the raw uploaded bytes) is deliberately excluded: retrieval never reads it, and
it's the bulk of the size.

---

## Step 1 — get the DB URL into a shell variable

```
DB=$(sudo -u maize /opt/maize/venv/bin/python -c "import os; from dotenv import load_dotenv; load_dotenv('/opt/maize/.env'); print(os.environ['DATABASE_URL'])")
```

**CHECK** — should print a `postgres://...` URL, not an error:

```
echo "$DB" | sed 's/:[^:@]*@/:****@/'
```

## Step 2 — confirm the TA is there and see how big it is

```
psql "$DB" -c "SELECT t.id, t.name, t.is_indexed, count(DISTINCT d.id) AS docs, count(c.id) AS chunks, pg_size_pretty(sum(length(c.chunk_text))::bigint) AS text_size FROM teaching_assistants t LEFT JOIN documents d ON d.ta_id=t.id LEFT JOIN document_chunks c ON c.ta_id=t.id WHERE t.id='iDYis09JtNUkyEJJ' GROUP BY t.id, t.name, t.is_indexed;"
```

**CHECK** — one row, `is_indexed = t`, non-zero docs and chunks. If `docs` is 0, stop and tell me:
the TA exists but was never indexed, which changes what we're reproducing.

## Step 3 — export the three tables to CSV in /tmp

```
mkdir -p /tmp/ta_pull && psql "$DB" -c "\copy (SELECT * FROM teaching_assistants WHERE id='iDYis09JtNUkyEJJ') TO '/tmp/ta_pull/ta.csv' CSV HEADER"
```

```
psql "$DB" -c "\copy (SELECT id, ta_id, filename, original_filename, display_name, file_type, file_size, storage_path, uploaded_at, doc_type, assignment_number, instructional_unit_number, instructional_unit_label, metadata_extracted, extraction_metadata, content_title, updated_at, last_indexed_at, doc_role, doc_role_provenance, bm25_tsvector, doc_category, summary, summary_embedding, full_text FROM documents WHERE ta_id='iDYis09JtNUkyEJJ' ORDER BY id) TO '/tmp/ta_pull/documents.csv' CSV HEADER"
```

```
psql "$DB" -c "\copy (SELECT * FROM document_chunks WHERE ta_id='iDYis09JtNUkyEJJ' ORDER BY id) TO '/tmp/ta_pull/chunks.csv' CSV HEADER"
```

## Step 4 — bundle it

```
tar -czf /tmp/ta_iDYis09JtNUkyEJJ.tar.gz -C /tmp/ta_pull ta.csv documents.csv chunks.csv && ls -lh /tmp/ta_iDYis09JtNUkyEJJ.tar.gz
```

**CHECK** — expect single-digit MB. If it's over ~200MB, stop and tell me the size before copying.

## Step 5 — pull it down (run this on your Mac, not the VPS)

```
scp root@getmaize.ai:/tmp/ta_iDYis09JtNUkyEJJ.tar.gz "/Users/simonkleffner/Desktop/Maize TA Master/App Dev/Maize-Blueprint-V2/scratchpad/"
```

Then tell me it's landed — I'll do the local import, remapping document and chunk ids so nothing
collides with the existing local rows (local is at doc id 165 / chunk id 15822).

## Step 6 — clean up the VPS

```
rm -rf /tmp/ta_pull /tmp/ta_iDYis09JtNUkyEJJ.tar.gz
```

---

## Rollback

Nothing on prod was changed, so there is nothing to roll back — Step 6 is just tidying /tmp.

On the local side, if the import goes wrong the undo is three deletes in dependency order:

```
docker exec maize_postgres_dev psql -U maize_dev -d maize_ta_dev -c "DELETE FROM document_chunks WHERE ta_id='iDYis09JtNUkyEJJ'; DELETE FROM documents WHERE ta_id='iDYis09JtNUkyEJJ'; DELETE FROM teaching_assistants WHERE id='iDYis09JtNUkyEJJ';"
```

## If Step 1 fails

`DATABASE_URL` may be set in the systemd unit rather than `.env`. In that case:

```
sudo systemctl show maize -p Environment | tr ' ' '\n' | grep -i DATABASE_URL
```
