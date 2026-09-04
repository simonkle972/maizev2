"""
Durable cleanup of external vendor state after a local delete.

Deleting an account or a TA has to reach three places we do not control —
Stripe, Auth0, and the filesystem — and the original code called all three
*before* the database commit. That made every failure destructive in one
direction or the other: an error at commit time left a live account whose
documents, vector index, subscription and login had already been destroyed,
while a failed Stripe call left a subscription billing forever behind a logged
warning nobody reads.

This module inverts that. `enqueue_*` records the obligations as rows in the
same transaction that deletes the local data, so:

  * if that transaction rolls back, nothing was destroyed and nothing is owed;
  * once it commits, every owed cleanup is recorded durably and survives a
    crashed request, a failed API call, or a process restart.

`run_for_origin` makes the immediate attempt inline with the request;
`process_pending` — wired to cron via `flask process-vendor-deletions` — retries
with backoff until each one succeeds. Every executor is idempotent: a vendor
object that is already gone counts as success, so retrying is always safe.
"""

import logging
import os
import shutil
from datetime import datetime, timedelta

from config import Config
from models import db, VendorDeletion

logger = logging.getLogger(__name__)

TARGET_STRIPE_SUBSCRIPTION = 'stripe_subscription'
TARGET_STRIPE_CUSTOMER = 'stripe_customer'
TARGET_AUTH0_USER = 'auth0_user'
TARGET_FILESYSTEM_PATH = 'filesystem_path'

# Minutes to wait before the Nth retry. The last value is the steady-state
# interval — we never stop retrying, because "stopped retrying" on a Stripe
# subscription means "kept billing a deleted account".
_BACKOFF_MINUTES = [5, 15, 60, 360, 1440]

# Attempts after which a row is reported as needing a human. It keeps retrying.
STUCK_AFTER_ATTEMPTS = 5


def _allowed_roots():
    """Directories a queued rmtree is permitted to touch.

    Paths are executed later, from a different process, off a value read out of
    the database — so containment is checked at execution time rather than
    trusted from the row.
    """
    return [
        os.path.abspath(Config.CHROMA_DB_PATH),
        os.path.abspath('data/courses'),
    ]


def enqueue(target, external_id, origin):
    """Queue one cleanup. Call inside the deleting transaction; do not commit."""
    if not external_id:
        return None
    row = VendorDeletion(target=target, external_id=str(external_id), origin=origin)
    db.session.add(row)
    return row


def enqueue_ta_cleanup(ta, origin):
    """Queue every external cleanup owed for one TA.

    The subscription is queued whenever an id exists, regardless of
    `requires_billing` — if Stripe has a subscription for this TA then Stripe
    will keep billing it, and the flag is not what decides that.
    """
    enqueue(TARGET_STRIPE_SUBSCRIPTION, ta.stripe_subscription_id, origin)
    enqueue(TARGET_FILESYSTEM_PATH,
            os.path.abspath(os.path.join(Config.CHROMA_DB_PATH, ta.id)), origin)
    enqueue(TARGET_FILESYSTEM_PATH,
            os.path.abspath(f'data/courses/{ta.id}'), origin)


def _stripe_gone(exc):
    """True when a Stripe error means the object is already deleted."""
    if getattr(exc, 'code', None) == 'resource_missing':
        return True
    return 'No such' in str(exc)


def _execute_stripe_subscription(external_id):
    import stripe
    try:
        stripe.Subscription.delete(external_id)
    except stripe.error.InvalidRequestError as e:
        if not _stripe_gone(e):
            raise


def _execute_stripe_customer(external_id):
    import stripe
    try:
        stripe.Customer.delete(external_id)
    except stripe.error.InvalidRequestError as e:
        if not _stripe_gone(e):
            raise


def _execute_auth0_user(external_id):
    # Already idempotent: treats a 404 as success, raises on anything else.
    from auth_auth0 import delete_auth0_user
    delete_auth0_user(external_id)


def _execute_filesystem_path(external_id):
    path = os.path.abspath(external_id)
    if not any(path == root or path.startswith(root + os.sep) for root in _allowed_roots()):
        raise ValueError(f"refusing to delete path outside the allowed roots: {path}")
    if os.path.exists(path):
        shutil.rmtree(path)


_EXECUTORS = {
    TARGET_STRIPE_SUBSCRIPTION: _execute_stripe_subscription,
    TARGET_STRIPE_CUSTOMER: _execute_stripe_customer,
    TARGET_AUTH0_USER: _execute_auth0_user,
    TARGET_FILESYSTEM_PATH: _execute_filesystem_path,
}


def _is_due(row, now):
    if row.attempts == 0 or row.last_attempt_at is None:
        return True
    wait = _BACKOFF_MINUTES[min(row.attempts - 1, len(_BACKOFF_MINUTES) - 1)]
    return now - row.last_attempt_at >= timedelta(minutes=wait)


def _run_one(row):
    """Attempt one cleanup. Returns True on success. Never raises."""
    now = datetime.utcnow()
    row.attempts += 1
    row.last_attempt_at = now
    try:
        _EXECUTORS[row.target](row.external_id)
    except Exception as e:
        row.last_error = str(e)[:2000]
        logger.warning(
            f"vendor cleanup failed ({row.target}, {row.origin}, "
            f"attempt {row.attempts}): {e}"
        )
        return False

    row.completed_at = now
    row.last_error = None
    # Keep the record, drop the vendor identifier — see the model docstring.
    row.external_id = None
    logger.info(f"vendor cleanup done ({row.target}, {row.origin})")
    return True


def _run(rows):
    """Run rows, commit the outcomes, return (succeeded, failed)."""
    succeeded = failed = 0
    for row in rows:
        if _run_one(row):
            succeeded += 1
        else:
            failed += 1
    if rows:
        db.session.commit()
    return succeeded, failed


def run_for_origin(origin):
    """Immediate attempt at everything queued under `origin`.

    Call this AFTER the deleting transaction has committed. Whatever fails stays
    pending and is picked up by process_pending.
    """
    rows = (VendorDeletion.query
            .filter_by(origin=origin, completed_at=None)
            .order_by(VendorDeletion.id)
            .all())
    return _run(rows)


def process_pending(limit=200):
    """Retry every pending cleanup that is due. Used by the cron command."""
    now = datetime.utcnow()
    pending = (VendorDeletion.query
               .filter_by(completed_at=None)
               .order_by(VendorDeletion.id)
               .all())
    due = [row for row in pending if _is_due(row, now)][:limit]
    succeeded, failed = _run(due)
    stuck = [row for row in pending
             if row.completed_at is None and row.attempts >= STUCK_AFTER_ATTEMPTS]
    return succeeded, failed, stuck
