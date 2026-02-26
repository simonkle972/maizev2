from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from pgvector.sqlalchemy import Vector
from werkzeug.security import generate_password_hash, check_password_hash

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

class User(db.Model):
    """User model for authentication - supports professors, students, and admins."""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(256), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(16), nullable=False)  # 'professor', 'student', 'admin'
    first_name = db.Column(db.String(128), nullable=False)
    last_name = db.Column(db.String(128), nullable=False)
    institution_id = db.Column(db.Integer, db.ForeignKey('institutions.id'), nullable=True, index=True)
    stripe_customer_id = db.Column(db.String(128), nullable=True, unique=True)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login_at = db.Column(db.DateTime, nullable=True)
    auth0_sub = db.Column(db.String(128), unique=True, nullable=True, index=True)
    email_verified = db.Column(db.Boolean, default=False, nullable=False)

    # Relationships
    institution = db.relationship('Institution', backref='users')

    # Flask-Login required properties
    @property
    def is_authenticated(self):
        return True

    @property
    def is_anonymous(self):
        return False

    def get_id(self):
        return str(self.id)

    # Password methods
    def set_password(self, password):
        self.password_hash = generate_password_hash(password, method='pbkdf2:sha256')

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    # Computed properties for professors
    @property
    def active_ta_count(self):
        """Count of active (non-paused) TAs for billing calculation."""
        if self.role != 'professor':
            return 0
        return TeachingAssistant.query.filter_by(
            professor_id=self.id,
            is_active=True,
            is_paused=False,
            requires_billing=True
        ).count()

    @property
    def total_monthly_cost(self):
        """Calculate total monthly billing across all active TAs."""
        if self.role != 'professor':
            return 0.0
        from config import Config
        tas = TeachingAssistant.query.filter_by(
            professor_id=self.id,
            is_active=True,
            is_paused=False,
            requires_billing=True
        ).all()
        return sum(Config.BILLING_TIERS.get(ta.billing_tier, {}).get('price_monthly', 0) for ta in tas)

class Institution(db.Model):
    __tablename__ = 'institutions'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(256), nullable=False)
    customer_id = db.Column(db.String(64), nullable=True, unique=True)
    email_domain = db.Column(db.String(256), nullable=True)  # Default domain for institution (e.g., "harvard.edu")
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    teaching_assistants = db.relationship('TeachingAssistant', backref='institution', lazy='dynamic')


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
    institution_id = db.Column(db.Integer, db.ForeignKey('institutions.id'), nullable=True)
    last_activity_at = db.Column(db.DateTime, nullable=True)

    # Authentication & Billing fields
    professor_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True, index=True)
    billing_tier = db.Column(db.String(32), default='tier1')  # tier1/tier2/tier3
    max_students = db.Column(db.Integer, default=50)
    allow_anonymous_chat = db.Column(db.Boolean, default=False)
    stripe_subscription_id = db.Column(db.String(128), nullable=True)  # Per-TA subscription
    subscription_status = db.Column(db.String(32), nullable=True)  # 'active', 'paused', 'canceled'
    is_paused = db.Column(db.Boolean, default=False)  # Paused TAs don't bill and are unavailable
    paused_at = db.Column(db.DateTime, nullable=True)
    last_pause_action_at = db.Column(db.DateTime, nullable=True)  # Track last pause/resume for cooldown
    requires_billing = db.Column(db.Boolean, default=True)  # False for admin-created TAs

    # Relationships
    professor = db.relationship('User', backref='taught_tas', foreign_keys=[professor_id])
    documents = db.relationship('Document', backref='ta', lazy='dynamic', cascade='all, delete-orphan')
    sessions = db.relationship('ChatSession', backref='ta', lazy='dynamic', cascade='all, delete-orphan')

    # Computed properties
    @property
    def current_enrollment_count(self):
        """Count of enrolled students."""
        return Enrollment.query.filter_by(ta_id=self.id).count()

    @property
    def is_available(self):
        """Check if TA is available (active and not paused)."""
        return self.is_active and not self.is_paused

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
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_indexed_at = db.Column(db.DateTime, nullable=True)

class ChatSession(db.Model):
    __tablename__ = 'chat_sessions'

    id = db.Column(db.String(32), primary_key=True)
    ta_id = db.Column(db.String(32), db.ForeignKey('teaching_assistants.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True, index=True)  # NULL for anonymous
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_activity = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    active_context = db.Column(db.JSON, nullable=True)

    # Relationships
    user = db.relationship('User', backref='chat_sessions')
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


class IndexingJob(db.Model):
    """Tracks indexing jobs for resumption after container restarts."""
    __tablename__ = 'indexing_jobs'

    id = db.Column(db.Integer, primary_key=True)
    ta_id = db.Column(db.String(32), db.ForeignKey('teaching_assistants.id'), nullable=False, index=True)
    status = db.Column(db.String(32), default='pending')  # pending, running, completed, failed
    started_at = db.Column(db.DateTime, nullable=True)
    completed_at = db.Column(db.DateTime, nullable=True)
    last_processed_doc_id = db.Column(db.Integer, nullable=True)
    docs_processed = db.Column(db.Integer, default=0)
    total_docs = db.Column(db.Integer, default=0)
    chunks_created = db.Column(db.Integer, default=0)
    error_message = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Enrollment(db.Model):
    """Links students to TAs they are enrolled in."""
    __tablename__ = 'enrollments'

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    ta_id = db.Column(db.String(32), db.ForeignKey('teaching_assistants.id'), nullable=False, index=True)
    enrolled_at = db.Column(db.DateTime, default=datetime.utcnow)
    enrollment_token = db.Column(db.String(64), nullable=True)
    first_accessed_at = db.Column(db.DateTime, nullable=True)  # Track when student first opens TA

    # Relationships
    student = db.relationship('User', backref='enrollments')
    ta = db.relationship('TeachingAssistant', backref='enrollments')

    __table_args__ = (
        db.UniqueConstraint('student_id', 'ta_id', name='unique_student_ta'),
    )


class EnrollmentLink(db.Model):
    """Manages signup links with capacity tracking."""
    __tablename__ = 'enrollment_links'

    id = db.Column(db.Integer, primary_key=True)
    ta_id = db.Column(db.String(32), db.ForeignKey('teaching_assistants.id'), nullable=False, index=True)
    token = db.Column(db.String(64), unique=True, nullable=False, index=True)
    max_capacity = db.Column(db.Integer, nullable=False)  # From billing tier
    current_enrollments = db.Column(db.Integer, default=0)
    is_active = db.Column(db.Boolean, default=True)
    expires_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    # Relationships
    ta = db.relationship('TeachingAssistant', backref='enrollment_links')
    creator = db.relationship('User', backref='created_links', foreign_keys=[created_by])

    @property
    def is_full(self):
        """Check if enrollment capacity is reached."""
        return self.current_enrollments >= self.max_capacity

    @property
    def is_valid(self):
        """Check if link is still valid (active, not full, not expired)."""
        if not self.is_active or self.is_full:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True


class PasswordResetToken(db.Model):
    """Manages password reset tokens."""
    __tablename__ = 'password_reset_tokens'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    token = db.Column(db.String(64), unique=True, nullable=False, index=True)
    expires_at = db.Column(db.DateTime, nullable=False)
    is_used = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    user = db.relationship('User', backref='reset_tokens')

    @property
    def is_valid(self):
        """Check if token is still valid (not used and not expired)."""
        return not self.is_used and datetime.utcnow() < self.expires_at


class Auth0State(db.Model):
    """Store Auth0 OAuth state in PostgreSQL."""
    __tablename__ = 'auth0_states'

    state = db.Column(db.String(256), primary_key=True)
    data = db.Column(db.JSON, nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)


class Auth0Transaction(db.Model):
    """Store Auth0 OAuth transactions in PostgreSQL."""
    __tablename__ = 'auth0_transactions'

    nonce = db.Column(db.String(256), primary_key=True)
    data = db.Column(db.JSON, nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)
