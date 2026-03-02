"""
Shared Flask extensions to avoid circular imports.
"""
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address


def _get_professor_id():
    """Rate limit key: professor's user ID (falls back to IP)."""
    try:
        from flask_login import current_user
        if current_user and current_user.is_authenticated:
            return f"prof:{current_user.id}"
    except Exception:
        pass
    return get_remote_address()


limiter = Limiter(
    key_func=_get_professor_id,
    storage_uri="memory://",
    default_limits=[]  # No global limits; only applied per-route
)
