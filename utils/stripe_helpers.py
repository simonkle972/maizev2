"""
Stripe integration helpers for managing subscriptions and billing.
"""

import stripe
from config import Config

# Initialize Stripe with the appropriate secret key
stripe.api_key = Config.STRIPE_SECRET_KEY


def create_stripe_customer(user):
    """
    Create a Stripe customer for a user.

    Args:
        user: User object

    Returns:
        Tuple of (customer_id: str, error: str or None)
    """
    try:
        customer = stripe.Customer.create(
            email=user.email,
            name=f"{user.first_name} {user.last_name}",
            metadata={
                'user_id': user.id,
                'role': user.role,
                'institution_id': user.institution_id
            }
        )
        return customer.id, None
    except stripe.error.StripeError as e:
        return None, str(e)


def create_checkout_session(professor, tier, ta_name, course_name, system_prompt, base_url):
    """
    Create a Stripe Checkout session for TA subscription.

    Args:
        professor: Professor User object
        tier: Billing tier ('tier1', 'tier2', 'tier3')
        ta_name: Name of the TA
        course_name: Name of the course
        system_prompt: System prompt for the TA
        base_url: Base URL for success/cancel redirects

    Returns:
        Tuple of (checkout_url: str, error: str or None)
    """
    try:
        tier_config = Config.BILLING_TIERS.get(tier)
        if not tier_config:
            return None, "Invalid billing tier"

        # Ensure professor has a Stripe customer ID
        if not professor.stripe_customer_id:
            customer_id, error = create_stripe_customer(professor)
            if error:
                return None, error
            from models import db
            professor.stripe_customer_id = customer_id
            db.session.commit()

        # Create Checkout session
        session = stripe.checkout.Session.create(
            customer=professor.stripe_customer_id,
            payment_method_types=['card'],
            line_items=[{
                'price': tier_config['stripe_price_id'],
                'quantity': 1,
            }],
            mode='subscription',
            success_url=f"{base_url}/professor/ta/create/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{base_url}/professor/ta/create?canceled=true",
            metadata={
                'professor_id': professor.id,
                'ta_name': ta_name,
                'course_name': course_name,
                'system_prompt': system_prompt,
                'billing_tier': tier
            }
        )

        return session.url, None

    except stripe.error.StripeError as e:
        return None, str(e)


def create_ta_from_checkout(checkout_session_id):
    """
    Create a TA after successful Stripe Checkout.

    Args:
        checkout_session_id: Stripe Checkout Session ID

    Returns:
        Tuple of (ta: TeachingAssistant, enrollment_link: EnrollmentLink, error: str or None)
    """
    try:
        from models import db, TeachingAssistant, EnrollmentLink, User
        import secrets

        # Retrieve checkout session
        session = stripe.checkout.Session.retrieve(checkout_session_id)

        if session.payment_status != 'paid':
            return None, None, "Payment not completed"

        # Get subscription
        subscription_id = session.subscription
        subscription = stripe.Subscription.retrieve(subscription_id)

        # Extract metadata
        metadata = session.metadata
        professor_id = int(metadata['professor_id'])
        ta_name = metadata['ta_name']
        course_name = metadata['course_name']
        system_prompt = metadata['system_prompt']
        tier = metadata['billing_tier']

        tier_config = Config.BILLING_TIERS[tier]
        professor = User.query.get(professor_id)

        # Generate unique TA ID and slug
        ta_id = secrets.token_urlsafe(12)
        slug = generate_slug(ta_name)

        # Create TA
        ta = TeachingAssistant(
            id=ta_id,
            slug=slug,
            name=ta_name,
            course_name=course_name,
            system_prompt=system_prompt,
            professor_id=professor_id,
            institution_id=professor.institution_id,
            billing_tier=tier,
            max_students=tier_config['max_students'],
            stripe_subscription_id=subscription_id,
            subscription_status='active',
            requires_billing=True,
            is_paused=False,
            allow_anonymous_chat=False
        )
        db.session.add(ta)
        db.session.flush()

        # Generate enrollment link
        token = secrets.token_urlsafe(32)
        link = EnrollmentLink(
            ta_id=ta.id,
            token=token,
            max_capacity=ta.max_students,
            created_by=professor_id,
            is_active=True
        )
        db.session.add(link)
        db.session.commit()

        return ta, link, None

    except stripe.error.StripeError as e:
        return None, None, str(e)
    except Exception as e:
        return None, None, str(e)


def pause_ta_subscription(ta):
    """
    Pause TA and pause its Stripe subscription billing.

    Args:
        ta: TeachingAssistant object

    Returns:
        Tuple of (success: bool, error: str or None)
    """
    try:
        from models import db
        from datetime import datetime, timedelta

        # Check if already paused (prevents concurrent requests)
        if ta.is_paused:
            return False, "TA is already paused."

        # COOLDOWN CHECK: Prevent pause/resume more than once per 30 days
        if ta.last_pause_action_at:
            days_since_last_action = (datetime.utcnow() - ta.last_pause_action_at).days
            if days_since_last_action < 30:
                days_remaining = 30 - days_since_last_action
                return False, (
                    f"You can only pause/resume once per 30 days to prevent abuse. "
                    f"Please wait {days_remaining} more day(s) before pausing again."
                )

        if ta.stripe_subscription_id:
            # Pause the subscription instead of deleting it
            # This preserves the payment method for when we resume
            stripe.Subscription.modify(
                ta.stripe_subscription_id,
                pause_collection={'behavior': 'void'}
            )

        ta.is_paused = True
        ta.paused_at = datetime.utcnow()
        ta.last_pause_action_at = datetime.utcnow()  # Track action time for cooldown
        ta.subscription_status = 'paused'
        db.session.commit()

        return True, None

    except stripe.error.StripeError as e:
        db.session.rollback()  # Explicit rollback on Stripe error
        return False, str(e)
    except Exception as e:
        db.session.rollback()  # Explicit rollback on any error
        return False, str(e)


def cancel_ta_subscription(ta):
    """
    Cancel a TA's Stripe subscription completely (for deletion).
    Similar to pause_ta_subscription but does not update TA record.
    Used when deleting a TA entirely.

    Args:
        ta: TeachingAssistant object

    Returns:
        Tuple of (success: bool, error: str or None)
    """
    try:
        if ta.stripe_subscription_id:
            stripe.Subscription.delete(ta.stripe_subscription_id)

        return True, None

    except stripe.error.StripeError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def resume_ta_subscription(ta):
    """
    Resume paused TA and unpause its Stripe subscription.

    Args:
        ta: TeachingAssistant object

    Returns:
        Tuple of (success: bool, error: str or None)
    """
    try:
        from models import db
        from datetime import datetime, timedelta

        if not ta.stripe_subscription_id:
            return False, "No subscription ID found. Please contact support."

        # Check if already active (prevents concurrent requests)
        if not ta.is_paused:
            return False, "TA is already active."

        # EDGE CASE: Check for corrupted paused_at (is_paused=True but paused_at=None)
        if ta.is_paused and not ta.paused_at:
            # Data corruption - set paused_at to now and require 7 days
            ta.paused_at = datetime.utcnow()
            db.session.commit()
            return False, "TA pause timestamp was missing. Please wait 7 days before resuming."

        # MINIMUM PAUSE DURATION: Must be paused for at least 7 days (CHECK FIRST)
        if ta.paused_at:
            days_paused = (datetime.utcnow() - ta.paused_at).days
            if days_paused < 7:
                days_remaining = 7 - days_paused
                return False, (
                    f"TA must be paused for at least 7 days before resuming to prevent billing abuse. "
                    f"Currently paused for {days_paused} day(s). Please wait {days_remaining} more day(s)."
                )

        # COOLDOWN CHECK: Prevent pause/resume more than once per 30 days (CHECK SECOND)
        if ta.last_pause_action_at:
            days_since_last_action = (datetime.utcnow() - ta.last_pause_action_at).days
            if days_since_last_action < 30:
                days_remaining = 30 - days_since_last_action
                return False, (
                    f"You can only pause/resume once per 30 days to prevent abuse. "
                    f"Please wait {days_remaining} more day(s) before resuming."
                )

        # Unpause the subscription by removing pause_collection
        stripe.Subscription.modify(
            ta.stripe_subscription_id,
            pause_collection=''
        )

        ta.is_paused = False
        ta.paused_at = None
        ta.last_pause_action_at = datetime.utcnow()  # Track action time for cooldown
        ta.subscription_status = 'active'
        db.session.commit()

        return True, None

    except stripe.error.StripeError as e:
        db.session.rollback()  # Explicit rollback on Stripe error
        return False, str(e)
    except Exception as e:
        db.session.rollback()  # Explicit rollback on any error
        return False, str(e)


def create_customer_portal_session(professor, base_url):
    """
    Create a Stripe Customer Portal session for billing management.

    Args:
        professor: Professor User object
        base_url: Base URL for return redirect

    Returns:
        Tuple of (portal_url: str, error: str or None)
    """
    try:
        if not professor.stripe_customer_id:
            return None, "No Stripe customer ID found"

        session = stripe.billing_portal.Session.create(
            customer=professor.stripe_customer_id,
            return_url=f"{base_url}/professor/settings/billing"
        )

        return session.url, None

    except stripe.error.StripeError as e:
        return None, str(e)


def generate_slug(name):
    """
    Generate a URL-friendly slug from a name.

    Args:
        name: Name to slugify

    Returns:
        slug: URL-friendly string
    """
    import re
    slug = name.lower().strip()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[-\s]+', '-', slug)

    # Ensure uniqueness
    from models import TeachingAssistant
    base_slug = slug
    counter = 1
    while TeachingAssistant.query.filter_by(slug=slug).first():
        slug = f"{base_slug}-{counter}"
        counter += 1

    return slug
