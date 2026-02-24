# Billing Abuse Prevention (Issue #12)

**Status:** ✅ Complete

## Problem
Professors could exploit Stripe's `pause_collection` to avoid billing by pausing right before billing dates and resuming immediately after.

**Exploitation Scenario:**
```
Jan 1:  Subscribe ($9.99 for Jan 1-31) ✓ Charged
Jan 30: Pause TA
Feb 1:  Invoice voided ✗ Not charged
Feb 2:  Unpause TA
Mar 1:  Next charge ($9.99 for Mar 1-31) ✓ Charged

Result: Free service for February
```

## Solution
Combined two constraints:
1. **Minimum 7-day pause** - Can't resume if paused < 7 days ago
2. **30-day cooldown** - Can only pause/resume once per 30 days

## Implementation

### Database Changes
- Added `last_pause_action_at` field to `TeachingAssistant` model
- Migration: `5cff802fc147_add_last_pause_action_at_to_.py`

### Backend Logic (`utils/stripe_helpers.py`)

**Pause Function:**
- Checks if already paused (prevents concurrent requests)
- Enforces 30-day cooldown between pause/resume actions
- Sets `last_pause_action_at` timestamp
- Uses `pause_collection={'behavior': 'void'}` to pause Stripe billing

**Resume Function:**
- Checks if already active (prevents concurrent requests)
- Handles corrupted `paused_at` data
- **Enforces 7-day minimum pause duration** (checked FIRST)
- **Enforces 30-day cooldown** (checked SECOND)
- Sets `last_pause_action_at` timestamp
- Removes `pause_collection` to resume Stripe billing

**Check Order Matters:**
- 7-day minimum applies to EVERY resume attempt
- 30-day cooldown only applies if professor already completed a full pause/resume cycle
- If someone pauses and immediately tries to resume, they see the 7-day error (not the 30-day error)

### UX Improvements

**Pause Confirmation:**
- Updated confirmation dialog to warn: "Once paused, it must remain paused for at least 7 days before you can resume"
- Location: `templates/professor/manage_ta.html` line 212

**Error Messages:**
- 7-day minimum: "TA must be paused for at least 7 days before resuming to prevent billing abuse. Currently paused for {days_paused} day(s). Please wait {days_remaining} more day(s)."
- 30-day cooldown: "You can only pause/resume once per 30 days to prevent abuse. Please wait {days_remaining} more day(s) before resuming/pausing again."

## Files Modified
1. `models.py` - Added `last_pause_action_at` field
2. `utils/stripe_helpers.py` - Updated `pause_ta_subscription()` and `resume_ta_subscription()`
3. `templates/professor/manage_ta.html` - Updated pause confirmation message
4. Migration: `migrations/versions/5cff802fc147_add_last_pause_action_at_to_.py`

## Testing
- ✅ First pause works (NULL last_pause_action_at)
- ✅ Immediate resume blocked by 7-day minimum
- ✅ Resume after 7+ days succeeds
- ✅ Immediate pause after resume blocked by 30-day cooldown
- ✅ Corrupted paused_at handled gracefully
- ✅ Concurrent requests prevented
- ✅ Error messages show correct constraint

## Success Criteria Met
✅ No professor can pause/resume more than once per 30 days
✅ No professor can resume before 7 days
✅ Billing exploitation blocked
✅ Clear error messages explain restrictions
✅ Legitimate use cases (semester breaks) still work
