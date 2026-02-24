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
