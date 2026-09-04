# Database Management

## Migrations

**IMPORTANT:** All schema changes are managed through Flask-Migrate migrations. Never use `db.create_all()` - this has been removed from the codebase.

### Basic Migration Commands

```bash
# ALWAYS set environment file first
export DOTENV_PATH=.env.local

# After modifying models.py, create a migration
flask db migrate -m "Description of changes"

# Review the generated migration file in migrations/versions/

# Apply migrations
flask db upgrade

# View current migration version
flask db current

# View migration history
flask db history

# Rollback last migration
flask db downgrade
```

### Migration Workflow

1. Modify `models.py`
2. Run `flask db migrate -m "Description"`
3. Review generated migration file
4. Test locally: `flask db upgrade`
5. Commit both `models.py` and the migration file
6. Deploy to production (migration runs automatically)

### Adding a New Model Field

1. Edit `models.py` to add the column
2. Create migration: `flask db migrate -m "Add field_name to Model"`
3. Review generated migration in `migrations/versions/`
4. Test locally: `flask db upgrade`
5. Deploy to production (migration runs automatically via deploy script)

## Docker Database Management

### Local Development

```bash
# Start PostgreSQL container
docker-compose up -d postgres

# Stop database
docker-compose down

# Reset database (WARNING: deletes all data)
docker-compose down -v
docker-compose up -d postgres
export DOTENV_PATH=.env.local
flask db upgrade

# Connect to database
docker exec -it maize_postgres_dev psql -U maize_dev -d maize_ta_dev
```

## Data Model

### Hierarchy

```
Institution
  └── TeachingAssistant (TA)
        ├── Documents (course materials)
        ├── DocumentChunks (embedded text chunks for RAG)
        ├── ChatSessions
        │     └── ChatMessages
        └── IndexingJobs (tracks async indexing progress)
```

### Key Relationships

- One institution can have many TAs
- Each TA has isolated documents and chat sessions
- Document chunks are created during indexing and contain pgvector embeddings
- IndexingJobs track progress for resumption after restarts

### Authentication Sessions

**Parallel Session Architecture**:
- Students: `session['_student_id']` (custom implementation in `auth_student.py`)
- Professors/Admin: `session['_user_id']` (Flask-Login)

These sessions can coexist on the same device without interference.

### Important Model Details

**Teaching Assistant IDs**:
- Type: `String(32)` (NOT integers!)
- Use `<ta_id>` in routes, never `<int:ta_id>`

## Connection Pooling

```python
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 600,
    "pool_pre_ping": True,  # Prevents "server has gone away" errors
    "pool_timeout": 60,
    "pool_size": 5,
    "max_overflow": 10,
    "connect_args": {
        "keepalives": 1,  # Critical for VPS reliability
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5,
    }
}
```

This prevents database connection issues in production.

## Backups and Data Deletion

Production runs on **Vultr Managed PostgreSQL (Business tier)**, which takes automatic
off-site backups with a **14-day point-in-time recovery window**. Backups are managed
entirely in the Vultr console — there is no `pg_dump` cron or backup script in this repo.

PITR retention is set by plan tier, so this number changes if the plan does:

| plan | PITR window |
|---|---|
| Hobbyist | none |
| Startup | 2 days |
| **Business (current)** | **14 days** |
| Premium | 30 days |

Individual backups can be deleted from the console. That is a blunt instrument — it
destroys a restore point for the whole database, not one account — so treat it as an
escape hatch for an insistent erasure request, not routine practice.

### What this means for account deletion

`professor.delete_account` removes local rows immediately and irreversibly, and queues
Stripe / Auth0 / filesystem cleanup in the same transaction (see
`utils/vendor_deletion.py`). But **copies of the deleted data survive in backups for up
to 14 days** before ageing out.

The accurate statement to make to a user, and the one any privacy policy should match:

> Live data is removed immediately and irreversibly. Copies persist in encrypted
> database backups for up to 14 days, after which they age out automatically.

Do not claim immediate erasure everywhere. Backups cannot be selectively edited, which
is a recognised exception — but only if the window is stated rather than implied away.

### ⚠️ Restoring a backup resurrects deleted accounts

This is the failure mode nobody anticipates. Restoring a snapshot taken **before** a
deletion brings back that professor's user row, TAs, documents and chat history — while:

- their Stripe customer and subscription are **already permanently cancelled**
- their Auth0 user is **already permanently deleted**
- the restored `vendor_deletions` rows come back marked `completed_at`, so
  `flask process-vendor-deletions` will **not** re-run anything

The result is a half-deleted account: local data alive, external identity destroyed, and
no queue entry recording that anything is owed. Nothing detects this automatically.

**Any restore therefore requires a manual pass to re-apply deletions that happened after
the snapshot was taken.** Before restoring, list the deletions in the window:

```sql
SELECT origin, target, completed_at
FROM vendor_deletions
WHERE completed_at > '<snapshot timestamp>'
ORDER BY completed_at;
```

`origin` is one of `account_delete:user=<id>`, `ta_delete:ta=<id>` or
`admin_ta_delete:ta=<id>`, so it identifies exactly which accounts and TAs need
deleting again once the restore completes.
