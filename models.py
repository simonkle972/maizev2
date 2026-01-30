from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from pgvector.sqlalchemy import Vector

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

class TeachingAssistant(db.Model):
    __tablename__ = 'teaching_assistants'
    
    id = db.Column(db.String(32), primary_key=True)
    slug = db.Column(db.String(64), unique=True, nullable=False, index=True)
    name = db.Column(db.String(256), nullable=False)
    course_name = db.Column(db.String(256), nullable=False)
    system_prompt = db.Column(db.Text, nullable=False, default="You are a helpful teaching assistant for this course. Help students understand course concepts by explaining clearly and guiding them through problems without giving direct answers.")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_indexed = db.Column(db.Boolean, default=False)
    indexed_at = db.Column(db.DateTime, nullable=True)
    document_count = db.Column(db.Integer, default=0)
    schema_version = db.Column(db.String(16), default="1.0")
    is_active = db.Column(db.Boolean, default=True)
    indexing_status = db.Column(db.String(32), default=None)
    indexing_error = db.Column(db.Text, nullable=True)
    indexing_progress = db.Column(db.Integer, default=0)
    
    documents = db.relationship('Document', backref='ta', lazy='dynamic', cascade='all, delete-orphan')
    sessions = db.relationship('ChatSession', backref='ta', lazy='dynamic', cascade='all, delete-orphan')

class Document(db.Model):
    __tablename__ = 'documents'
    
    id = db.Column(db.Integer, primary_key=True)
    ta_id = db.Column(db.String(32), db.ForeignKey('teaching_assistants.id'), nullable=False)
    filename = db.Column(db.String(512), nullable=False)
    original_filename = db.Column(db.String(512), nullable=False)
    display_name = db.Column(db.String(512), nullable=True)
    file_type = db.Column(db.String(32), nullable=False)
    file_size = db.Column(db.Integer, nullable=False)
    storage_path = db.Column(db.String(1024), nullable=False)
    file_content = db.Column(db.LargeBinary, nullable=True)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    doc_type = db.Column(db.String(64), nullable=True)
    assignment_number = db.Column(db.String(32), nullable=True)
    instructional_unit_number = db.Column(db.Integer, nullable=True)
    instructional_unit_label = db.Column(db.String(64), nullable=True)
    metadata_extracted = db.Column(db.Boolean, default=False)
    extraction_metadata = db.Column(db.JSON, nullable=True)
    content_title = db.Column(db.String(512), nullable=True)

class ChatSession(db.Model):
    __tablename__ = 'chat_sessions'
    
    id = db.Column(db.String(32), primary_key=True)
    ta_id = db.Column(db.String(32), db.ForeignKey('teaching_assistants.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_activity = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    messages = db.relationship('ChatMessage', backref='session', lazy='dynamic', cascade='all, delete-orphan', order_by='ChatMessage.created_at')

class ChatMessage(db.Model):
    __tablename__ = 'chat_messages'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(32), db.ForeignKey('chat_sessions.id'), nullable=False)
    role = db.Column(db.String(16), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    sources = db.Column(db.JSON, nullable=True)

class DocumentChunk(db.Model):
    __tablename__ = 'document_chunks'
    
    id = db.Column(db.Integer, primary_key=True)
    ta_id = db.Column(db.String(32), db.ForeignKey('teaching_assistants.id'), nullable=False, index=True)
    document_id = db.Column(db.Integer, db.ForeignKey('documents.id'), nullable=False)
    chunk_index = db.Column(db.Integer, nullable=False)
    chunk_text = db.Column(db.Text, nullable=False)
    chunk_context = db.Column(db.String(256), nullable=True)
    doc_type = db.Column(db.String(64), nullable=True)
    assignment_number = db.Column(db.String(32), nullable=True)
    instructional_unit_number = db.Column(db.Integer, nullable=True)
    instructional_unit_label = db.Column(db.String(64), nullable=True)
    file_name = db.Column(db.String(512), nullable=True)
    embedding = db.Column(Vector(1536), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
