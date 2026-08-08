# PRD: Deferred Billing — Pay to Publish, Not to Create

## Overview

Move the billing step from before TA creation to after TA creation but before publishing. Professors should be able to create a TA, upload documents, and configure settings without a subscription. Payment is required only to publish the TA (i.e., generate a student-accessible link).

## Problem

The current flow requires professors to select a billing tier and start a subscription before they can create a TA. This asks for financial commitment before the professor has seen any value from the product. This is a conversion risk, especially for the target market of cost-conscious instructors at underserved institutions.

## New Flow

### Current Flow
1. Professor signs up
2. Professor selects billing tier and subscribes (Stripe)
3. Professor creates TA (name, settings)
4. Professor uploads course materials
5. TA is live — student link is active

### New Flow
1. Professor signs up
2. Professor creates TA (name, settings) — **no payment required**
3. Professor uploads course materials — **no payment required**
4. Professor tests TA in-portal *(future feature, not part of this PRD but the flow should accommodate it here)*
5. Professor selects billing tier and subscribes (Stripe) — **payment happens here**
6. Professor publishes TA — student link becomes active

### Key Principle

**Creation and configuration are free. Publishing requires a subscription.** The TA exists in a draft/unpublished state until the professor pays and explicitly publishes.

---

## TA Lifecycle States

Introduce a state model for TAs:

| State | Description | Student Access | Professor Can Edit | Subscription Required |
|---|---|---|---|---|
| `draft` | Created, may or may not have documents uploaded. Not accessible to students. | No | Yes | No |
| `active` | Published. Student link is live. Subscription is active. | Yes | Yes | Yes |
| `paused` | Subscription lapsed or professor manually unpublished. Student link returns an unavailable message. | No | Yes | No (to remain paused) |

### State Transitions

- `draft` → `active`: Professor selects tier, completes Stripe subscription, and publishes.
- `active` → `paused`: Subscription expires/cancels, or professor manually unpublishes.
- `paused` → `active`: Professor resubscribes or resumes subscription and republishes.
- `active` → `draft`: Should not happen. Once published, a TA moves to `paused` if deactivated, not back to `draft`.
- `draft` → `draft`: Professor continues editing and uploading materials. No state change.

---

## Database Changes

### Changes to TA Table

Add or modify the following columns:

| Column | Type | Notes |
|---|---|---|
| status | ENUM: `draft`, `active`, `paused` | Default `draft`. Replaces any existing boolean like `is_published` |
| published_at | TIMESTAMP | Nullable. Set when TA first moves to `active` |
| subscription_id | FK or VARCHAR | Nullable. Reference to Stripe subscription. Null while in `draft` |
| tier | ENUM or VARCHAR | Nullable. Set when professor selects a tier during the publish flow. Null while in `draft` |

### Migration Notes

- Existing TAs with active subscriptions: migrate to `status = 'active'`.
- Any TAs without subscriptions (edge case): migrate to `status = 'draft'`.

### Design Consideration for Future Tier Upgrades

A follow-up feature will allow professors to change tiers on an active TA (e.g., Starter → Standard when enrollment grows). To support this:

- `tier` on the TA table must be mutable, not permanently tied to a static subscription type.
- Stripe subscription must be modifiable via the Stripe API (plan change with proration).
- Do not hardcode tier-to-subscription mappings in a way that prevents tier changes later.
- For this PRD, `tier` is set once during the publish flow. The upgrade feature will modify it later.

---

## UI Changes

### TA Dashboard / List View

- Show TA status as a badge or label next to each TA: `Draft`, `Active`, or `Paused`.
- Draft TAs: show a prominent "Publish" button.
- Active TAs: show "Published" indicator and the student link with a copy button.
- Paused TAs: show a "Reactivate" button.

### TA Creation Flow

- Remove the billing/tier selection step entirely from TA creation.
- Professor goes straight from "Create TA" to naming it, configuring settings, and uploading documents.
- No mention of pricing during creation. The experience should feel completely free.

### Publish Flow (New)

When a professor clicks "Publish" on a draft TA:

1. **Tier selection screen:** Show the three pricing tiers with student caps. The professor selects the tier that fits their course. Class size is not known at this point since it is not collected during creation, so the professor chooses based on their own knowledge of enrollment.
2. **Stripe checkout:** Professor enters payment info and starts subscription. Use Stripe Checkout or the existing Stripe integration. Subscription is tied to this specific TA.
3. **Confirmation and publish:** On successful payment, set TA status to `active`, store `subscription_id` and `tier`, set `published_at`, and generate/activate the student-accessible link.
4. **Show the student link:** Immediately display the student link with a "Copy link" button. This is the payoff moment — make it feel like an accomplishment.

### Publish Button Behavior

- The "Publish" button is always visible and clickable on draft TAs.
- Do not grey it out or disable it based on whether documents have been uploaded. A professor may want to publish first and add materials later. Do not block this.
- The only prerequisite for clicking "Publish" is that the TA exists.

### Student Link Behavior by State

| TA State | Student visits link | Behavior |
|---|---|---|
| `draft` | Link does not exist or returns 404 | No public URL generated until first publish |
| `active` | Normal TA experience | Chat interface loads |
| `paused` | Unavailable message | "This teaching assistant is currently unavailable. Please contact your instructor." |

---

## Stripe Integration Changes

### Current Behavior
- Subscription is created before TA creation.
- TA creation is gated behind active subscription.

### New Behavior
- No Stripe interaction during TA creation.
- Subscription is created during the publish flow.
- Maintain existing 1:1 TA-to-subscription relationship if that is the current model.

### Webhook Handling

Handle these Stripe webhook events:

- **`invoice.payment_succeeded`**: If TA is `paused` due to prior payment failure, move back to `active`.
- **`invoice.payment_failed`**: Move TA to `paused`. Student link shows unavailable message. Send professor an email notification.
- **`customer.subscription.deleted`**: Move TA to `paused`. Covers cancellation, expiry, etc.

---

## Cost Exposure from Draft TAs

Professors in `draft` state incur indexing/embedding costs when they upload documents without an active subscription. This is an accepted tradeoff to improve conversion.

### Guardrails (Optional — Not Required for v1)

- **Soft document cap for drafts:** If cost becomes a concern, limit draft TAs to a reasonable ceiling (e.g., 20 files or 200 pages). Show a message: "Publish your TA to upload additional materials." Only implement if abuse or cost becomes a problem.
- **Draft expiry:** Consider auto-deleting draft TAs inactive for 90+ days to reclaim storage. Low priority.

---

## Implementation Priority

1. **Add `status` field to TA table and migrate existing TAs** — foundation for everything
2. **Remove billing gate from TA creation flow** — professors create and upload freely
3. **Build the publish flow** — tier selection → Stripe checkout → set active → show student link
4. **Student link behavior by state** — 404 for draft, chat for active, unavailable message for paused
5. **Stripe webhook handling** — payment failure and cancellation move TA to paused
6. **TA dashboard UI updates** — status badges, publish button, reactivate button
7. **Reactivation flow** — paused → active via resubscription

---

## Success Metrics

Track from day one:

- **Draft-to-published conversion rate:** Percentage of created TAs that get published. This is the primary metric for evaluating this change.
- **Time from creation to publish:** Minutes suggests a smooth flow. Days suggests professors need a nudge or aren't convinced.
- **Abandoned drafts:** Number of TAs created but never published. High volume is acceptable (low-friction creation is working) as long as conversion rate is healthy.
- **Documents uploaded before publish:** Indicates how much testing professors do before committing. Useful signal for when the in-portal testing feature ships.
