from app import app, db
from sqlalchemy import text

with app.app_context():
    try:
        db.session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        db.session.commit()
        print("pgvector extension enabled")
    except Exception as e:
        db.session.rollback()
        print(f"Note: pgvector extension - {e}")
    
    db.create_all()
    
    columns_to_add = [
        ("teaching_assistants", "indexing_status", "VARCHAR(32)"),
        ("teaching_assistants", "indexing_error", "TEXT"),
        ("teaching_assistants", "indexing_progress", "INTEGER DEFAULT 0"),
        ("documents", "file_content", "BYTEA"),
    ]
    
    for table, column, col_type in columns_to_add:
        try:
            db.session.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"))
            db.session.commit()
            print(f"Added column {column} to {table}")
        except Exception as e:
            db.session.rollback()
            if "already exists" in str(e).lower() or "duplicate column" in str(e).lower():
                print(f"Column {column} already exists in {table}")
            else:
                print(f"Note: {column} - {e}")
    
    try:
        db.session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding 
            ON document_chunks USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """))
        db.session.commit()
        print("Vector index created on document_chunks")
    except Exception as e:
        db.session.rollback()
        if "already exists" in str(e).lower():
            print("Vector index already exists")
        else:
            print(f"Note: vector index - {e}")
    
    print('Database initialized')
