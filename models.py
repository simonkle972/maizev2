from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.dialects.postgresql import TSVECTOR
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
    onboarding_complete = db.Column(db.Boolean, default=False, nullable=False)
    institution_verified = db.Column(db.Boolean, default=False, nullable=False, server_default='false')
    verification_domain = db.Column(db.String(256), nullable=True)

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
        """Calculate total monthly billing across all active (published, non-paused) TAs."""
        if self.role != 'professor':
            return 0.0
        from config import Config
        tas = TeachingAssistant.query.filter(
            TeachingAssistant.professor_id == self.id,
            TeachingAssistant.requires_billing == True,
            TeachingAssistant.status == 'active',  # Only published, non-paused TAs
        ).all()
        return sum(Config.BILLING_TIERS.get(ta.billing_tier, {}).get('price_monthly', 0) for ta in tas)

class Institution(db.Model):
    __tablename__ = 'institutions'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(256), nullable=False)
    customer_id = db.Column(db.String(64), nullable=True, unique=True)
    email_domain = db.Column(db.String(256), nullable=True)  # Legacy single-domain field; use InstitutionDomain for new records
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_from_dataset = db.Column(db.Boolean, default=False, nullable=False, server_default='false')
    country = db.Column(db.String(128), nullable=True)
    alpha_two_code = db.Column(db.String(2), nullable=True)
    state_province = db.Column(db.String(128), nullable=True)
    web_pages = db.Column(db.JSON, nullable=True)

    teaching_assistants = db.relationship('TeachingAssistant', backref='institution', lazy='dynamic')


class InstitutionDomain(db.Model):
    """One-to-many domains per institution for email-based verification."""
    __tablename__ = 'institution_domains'

    id = db.Column(db.Integer, primary_key=True)
    institution_id = db.Column(db.Integer, db.ForeignKey('institutions.id'), nullable=False, index=True)
    domain = db.Column(db.String(256), nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    institution = db.relationship('Institution', backref='domains')


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
    # Per-doc failures from the last indexing run — list of {doc_id, filename, error}.
    # Distinct from `indexing_error` (which is a single TA-level error message for hard failures).
    # Populated by run_indexing_task; consumed by the manage_ta UI to surface per-doc failures.
    indexing_warnings = db.Column(db.JSON, nullable=True)
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
    status = db.Column(db.String(16), nullable=False, default='draft', server_default='draft')  # 'draft' | 'active' | 'paused'
    published_at = db.Column(db.DateTime, nullable=True)

    # Phase 1: admin-only feature. Toggle is hidden from professors; only admin-created TAs can have it on.
    image_upload_enabled = db.Column(db.Boolean, default=False, nullable=False, server_default='false')

    # Phase A retrieval refactor — Stage 2B (gap analysis 2026-05-22 + research
    # 2026-05-22). Per-TA configurable document categories. JSON array of
    # {slug, label} objects. Replaces the global doc_role enum from Stage 2 with
    # a per-TA controlled vocabulary. Default seeded at TA creation; professor
    # editable. Documents reference categories by slug.
    doc_categories = db.Column(db.JSON, nullable=True)

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
        """Check if TA is available (published and not paused)."""
        return self.status == 'active'

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
    # Phase A retrieval refactor — Stage 2 (gap analysis 2026-05-22). DEPRECATED
    # in Stage 2B (research 2026-05-22): doc_role's fixed 6-value enum was the
    # same enum-rigidity that caused the Type A failure. Column survives for
    # backwards-compat / rollback; no longer load-bearing. See doc_category below.
    doc_role = db.Column(db.String(32), nullable=True)
    # DEPRECATED with doc_role in Stage 2B. Survives for backwards-compat.
    doc_role_provenance = db.Column(db.JSON, nullable=True)
    # BM25 (full-text search) tsvector over the document's extracted text.
    # Substrate for the hybrid Stage 1 retrieval (BM25 + dense + RRF).
    # Indexed with GIN — see the migration. Populated by the indexing pipeline.
    bm25_tsvector = db.Column(TSVECTOR, nullable=True)
    # Phase A retrieval refactor — Stage 2B (gap analysis 2026-05-22 +
    # research 2026-05-22). REPLACES doc_role as the primary classification axis
    # for retrieval. Stores the SLUG of one of the parent TA's doc_categories.
    # The slug-vs-label split (Library Drift mitigation, arXiv 2605.19576) means
    # category renames don't orphan classified documents. Auto-classified at
    # upload via LLM from the TA's configured list; professor overrides via UI.
    doc_category = db.Column(db.String(64), nullable=True)

    # Phase B Stage B10 (architecture audit 2026-05-23, cross-cutting finding #3).
    # Per-doc LLM-generated summary + its embedding. Sets up the future refactor
    # of hybrid_doc_search: replace BM25 + dense + filename RRF + Stage 5 short-
    # circuit with summary-cosine + LLM tiebreaker (~300 lines deletable, biggest
    # single retrieval-side simplification in the audit). Today these columns
    # are populated at index time + via backfill but NOT yet read by retrieval —
    # ships as indexing-only infrastructure pending the focused retrieval-side
    # session that wires summary_embedding into doc-routing.
    summary = db.Column(db.Text, nullable=True)
    summary_embedding = db.Column(Vector(1536), nullable=True)


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

    # Up to MAX_IMAGES_PER_TURN attached images per user message, in display order.
    # See ChatMessageImage below. Replaces the old single-image image_data/image_mime columns
    # (migrated 2026-04-28).
    images = db.relationship(
        'ChatMessageImage',
        backref='message',
        lazy='select',
        order_by='ChatMessageImage.order',
        cascade='all, delete-orphan',
    )


class ChatMessageImage(db.Model):
    __tablename__ = 'chat_message_images'

    id = db.Column(db.Integer, primary_key=True)
    message_id = db.Column(db.Integer, db.ForeignKey('chat_messages.id'), nullable=False, index=True)
    image_data = db.Column(db.LargeBinary, nullable=False)
    image_mime = db.Column(db.String(64), nullable=True)
    order = db.Column(db.Integer, nullable=False, default=0)  # 0-indexed display order within the message
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class DocumentChunk(db.Model):
    __tablename__ = 'document_chunks'
    
    id = db.Column(db.Integer, primary_key=True)
    ta_id = db.Column(db.String(32), db.ForeignKey('teaching_assistants.id'), nullable=False, index=True)
    document_id = db.Column(db.Integer, db.ForeignKey('documents.id'), nullable=False)
    chunk_index = db.Column(db.Integer, nullable=False)
    chunk_text = db.Column(db.Text, nullable=False)
    chunk_context = db.Column(db.String(256), nullable=True)
    # Phase B Stage B8 (2026-05-25). Multi-level structural path for the chunk
    # within its parent doc: ["Section II", "Part b"] / ["Slide 3"] / ["Page 5"] / [].
    # Replaces single-header chunk_context as the load-bearing structural-navigation
    # signal. chunk_context kept for backwards-compat with B15 structural injection
    # until B15 is deleted in a follow-up cleanup commit.
    section_path = db.Column(db.JSON, nullable=True)
    doc_type = db.Column(db.String(64), nullable=True)
    assignment_number = db.Column(db.String(32), nullable=True)
    instructional_unit_number = db.Column(db.Integer, nullable=True)
    instructional_unit_label = db.Column(db.String(64), nullable=True)
    file_name = db.Column(db.String(512), nullable=True)
    # Phase A retrieval refactor — Stage 2 (DEPRECATED in Stage 2B). Survives
    # for backwards-compat; no longer load-bearing. See doc_category below.
    doc_role = db.Column(db.String(32), nullable=True)
    # Phase A retrieval refactor — Stage 2B. Denormalized copy of
    # Document.doc_category (the SLUG) so the retriever can filter chunks
    # without joining back to documents. Synced via the metadata-edit PATCH
    # routes and the indexing pipeline.
    doc_category = db.Column(db.String(64), nullable=True)
    # Phase B Stage B9 (2026-05-25). Anthropic Contextual Retrieval — 1-2
    # sentence LLM-generated context blob situating the chunk within its parent
    # doc. Prepended to the embedded text at index time; the raw chunk_text
    # above is unchanged and remains the display-time return value. Stored
    # separately so we can debug "why did this chunk match" and re-embed
    # without re-running the LLM. NULL on pre-B9 chunks until the backfill
    # script repopulates them. See attached_assets/maize-b9-contextual-
    # retrieval-plan.md and section 3.3 of maize-architecture-review-2026-05-23.md.
    contextual_prefix = db.Column(db.Text, nullable=True)
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
