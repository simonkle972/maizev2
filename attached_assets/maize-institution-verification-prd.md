# PRD: Institution Verification for Professor Accounts

## Overview

Implement email-based institution verification for professor accounts using the Hipo university-domains-list dataset. This feature ensures data integrity around institutional affiliations without adding friction to the signup or profile management flows.

## Problem

Currently, professors self-declare their institution with no validation. Anyone can claim to be affiliated with any university. This creates two problems: Maize cannot trust institutional data for analytics or sales purposes, and it undermines a potential differentiator against competitors (e.g., All Day TA) who have the same gap.

## Goals

- Verify professor-institution associations using .edu / institutional email domains
- Maintain zero-friction signup for professors with institutional emails
- Allow professors without institutional emails (Gmail, etc.) to still sign up, with an "unverified" status
- Surface verification status in a useful way (backend data, student-facing badge)
- Avoid hard-blocking any professor from signing up or associating with any institution

## Non-Goals

- Blocking signups based on email domain
- Building a custom university database from scratch
- Verifying student identities or institution associations
- Requiring institutional email as the primary account email

---

## Data Source

**Repository:** https://github.com/Hipo/university-domains-list  
**License:** MIT  
**File:** `world_universities_and_domains.json`  
**Hosted API (fallback/reference):** http://universities.hipolabs.com

### Data Structure

```json
{
  "alpha_two_code": "US",
  "country": "United States",
  "state-province": "Virginia",
  "domains": ["wcc.vccs.edu"],
  "name": "Wytheville Community College",
  "web_pages": ["http://www.wcc.vccs.edu/"]
}
```

### Implementation Notes on the Dataset

- **Store locally.** Download and seed the `world_universities_and_domains.json` into a database table on deploy. Do not depend on the hosted API at runtime.
- **Universities can have multiple domains.** The `domains` field is an array. Yale, for example, has `yale.edu` but professors may email from subdomains like `som.yale.edu`. Domain matching must account for this (see Matching Logic below).
- **The dataset is international.** Includes universities from 200+ countries. Not all use `.edu` TLDs — e.g., `.ac.uk` (UK), `.edu.au` (Australia). The matching logic should work with any domain, not just `.edu`.
- **The dataset may have gaps.** Some small or new institutions may be missing. The system must handle this gracefully (professor can still sign up and self-declare).
- **Periodic refresh.** Consider a mechanism (manual or scheduled) to pull updated versions of the JSON from GitHub periodically. Not urgent, but worth designing for.

---

## Database Changes

### New Table: `institutions`

Seed from the Hipo dataset. This replaces or augments any existing institution storage.

| Column | Type | Notes |
|---|---|---|
| id | UUID / PK | |
| name | VARCHAR | From dataset `name` field |
| country | VARCHAR | From dataset `country` field |
| alpha_two_code | VARCHAR(2) | From dataset `alpha_two_code` |
| state_province | VARCHAR | Nullable. From dataset `state-province` |
| web_pages | JSON/ARRAY | From dataset `web_pages` |
| is_from_dataset | BOOLEAN | `true` if seeded from Hipo data, `false` if manually created by a professor |
| created_at | TIMESTAMP | |

### New Table: `institution_domains`

One-to-many from `institutions`. Enables efficient domain lookups.

| Column | Type | Notes |
|---|---|---|
| id | UUID / PK | |
| institution_id | FK → institutions | |
| domain | VARCHAR | e.g., `yale.edu`, `wcc.vccs.edu` |
| created_at | TIMESTAMP | |

**Index:** Create an index on `domain` for fast lookups.

### Changes to Professor/User Table

Add the following columns (or modify existing institution association):

| Column | Type | Notes |
|---|---|---|
| institution_id | FK → institutions | Nullable. The institution the professor is associated with |
| institution_verified | BOOLEAN | Default `false`. `true` if email domain matches institution |
| verification_domain | VARCHAR | Nullable. The email domain that was matched for verification |

---

## Domain Matching Logic

This is the core of the verification system. Must handle subdomain matching.

### Algorithm

Given a professor's email (e.g., `john.smith@som.yale.edu`):

1. Extract the email domain: `som.yale.edu`
2. Generate candidate domains by progressively stripping subdomains:
   - `som.yale.edu`
   - `yale.edu`
3. Query `institution_domains` for any matching domain in the candidate list
4. If a match is found, return the associated institution(s)

### Edge Cases

- **Multiple institutions share a parent domain:** Unlikely but possible. If multiple institutions match, return all matches and let the professor choose.
- **Professor email is Gmail/Outlook/etc.:** No match. Professor remains unverified. This is fine.
- **Professor's institution exists in dataset but they use a personal email:** They can add an institutional email later (see Profile Settings flow) to verify.
- **Professor's institution is NOT in the dataset:** They can create a new institution entry (`is_from_dataset = false`). They remain unverified unless they later provide an email that matches a known domain.

---

## User Flows

### Flow 1: Signup (Modified)

**Current behavior:** Professor types institution name in a free-text field during signup. If it doesn't exist, they create it.

**New behavior:**

1. Professor enters their email address (already part of signup).
2. System extracts the email domain and runs the matching algorithm.
3. **If a match is found:**
   - Auto-suggest the matched institution(s) in the institution field: "Based on your email, are you affiliated with [Yale University]?"
   - If professor confirms: set `institution_id` to the matched institution, set `institution_verified = true`, store `verification_domain`.
   - If professor declines and picks a different institution: set `institution_id` to their choice, `institution_verified = false`.
4. **If no match is found:**
   - Professor types institution name as before with autocomplete against the `institutions` table.
   - If they find their institution: set `institution_id`, `institution_verified = false`.
   - If they don't find it: they can create a new institution entry. `institution_verified = false`, `is_from_dataset = false` on the new institution.
5. **Institution field is optional.** A professor can skip this entirely during signup.

### Flow 2: Profile Settings (Modified)

**Current behavior:** Professor can update their institution in profile settings.

**New behavior:**

1. Profile settings shows current institution and verification status.
2. **If unverified:** Show a prompt: "Verify your affiliation by adding your institutional email address."
   - Professor enters an institutional email (this is stored as a secondary/verification email, NOT replacing their primary login email).
   - System runs domain matching against the professor's current institution.
   - If the email domain matches: set `institution_verified = true`, store `verification_domain`.
   - If the email domain matches a *different* institution: inform the professor ("This email is associated with [Other University]. Would you like to switch your institution?") and let them choose.
   - If no match: inform the professor that the domain couldn't be verified. They remain unverified.
3. **If already verified:** Show the verified badge and the email that was used for verification. Allow them to change institution (which resets `institution_verified = false` and clears `verification_domain`).
4. **Changing institution:** If a professor changes their institution in settings, reset verification status. If their account email matches the new institution's domain, auto-verify. Otherwise, unverified.

### Flow 3: Institution Autocomplete (Both Signup and Settings)

- As the professor types in the institution field, autocomplete against the `institutions` table by name.
- Show country next to the name for disambiguation (e.g., "University of Melbourne — Australia").
- If the professor's email has already been matched to an institution, pin that institution at the top of the autocomplete results with a "Suggested based on your email" label.
- If no results match: show option to "Add [typed name] as a new institution."

---

## Student-Facing Changes

### Verified Instructor Badge

When a professor's `institution_verified = true`, display a small badge on their student-facing TA interface. Suggested placement: near the course title in the header area.

**Badge content:** "Verified [Institution Name] instructor" or a checkmark icon with the institution name.

**When unverified:** Do not show any badge. Do not show "unverified" — just omit it. No need to signal a negative.

---

## Admin / Internal Visibility

### Maize Admin Dashboard (for you internally)

- List of all institutions with course counts and verified/unverified professor counts.
- Ability to filter professors by verification status.
- Ability to manually verify a professor (override) if needed.
- Ability to manually add domains to an institution (for cases where the Hipo dataset is missing a subdomain).
- Flag for institutions created by professors (`is_from_dataset = false`) so you can review and potentially add them to the dataset or merge duplicates.

---

## Analytics Implications

With verification in place, analytics can be segmented:

- **Verified institutions only:** High-confidence data for sales and social proof. "Verified professors at X institutions use Maize."
- **All institutions (verified + self-reported):** Broader view, useful for internal analysis.
- This distinction should be carried through to any analytics dashboard or reporting you build later.

---

## Implementation Priority

1. **Database tables and seeding from Hipo dataset** — foundation for everything else
2. **Domain matching logic** — core utility function, reused across flows
3. **Signup flow modification** — auto-suggest institution based on email
4. **Profile settings flow** — ability to verify after the fact
5. **Institution autocomplete** — better UX for both flows
6. **Student-facing badge** — visible payoff of verification
7. **Admin dashboard additions** — internal tooling, lower urgency

---

## Technical Notes

- The Hipo dataset JSON is ~5MB. Seed it into the database on initial deploy or via a migration script. Parse the JSON, create `institutions` rows, and create `institution_domains` rows for each entry in the `domains` array.
- Domain matching should be a reusable utility function (e.g., `match_institution_by_email(email: str) -> Optional[Institution]`) since it's used in signup, profile settings, and potentially future flows.
- The autocomplete endpoint should support fuzzy matching on institution names (e.g., "yale" matches "Yale University"). Consider using `ILIKE` with `%` wildcards or a trigram index if performance becomes an issue at scale.
- For the verification email in profile settings: you do NOT need to send a verification/confirmation email with a link. The fact that the professor can receive email at that address is validated by the fact that they signed up with it or entered it. If you want extra security later, you can add email confirmation, but it's not necessary for v1.
