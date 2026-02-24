# Maize TA Documentation

This directory contains documentation for completed features, fixes, and architectural decisions.

## Structure

```
/docs/
  README.md                    # This file
  /completed/                  # Completed features and fixes
    billing-abuse-prevention.md
    post-testing-fixes.md
```

## Documentation Guidelines

### What Goes Here
- **Completed features** - Full implementation details, rationale, and testing
- **Major fixes** - Post-implementation documentation of significant bug fixes
- **Architecture decisions** - Why certain approaches were chosen
- **Reference material** - Information needed for future development

### What Doesn't Go Here
- **Active work** - Keep in plan files (`.claude/plans/`)
- **Code comments** - Belongs in the code itself
- **API docs** - Consider separate API documentation if needed
- **User guides** - Consider separate user documentation

### File Naming
- Use lowercase with hyphens: `billing-abuse-prevention.md`
- Be descriptive: `student-dashboard-redesign.md` not `issue-9.md`
- Group related docs in subdirectories when needed

### Document Structure
Each completed feature doc should include:
1. **Status** - ✅ Complete, ⚠️ Partial, etc.
2. **Problem** - What was the issue?
3. **Solution** - How was it solved?
4. **Implementation** - Key technical details
5. **Files Modified** - What changed?
6. **Testing** - How was it verified?

## Finding Information

- **Billing & Stripe**: See `completed/billing-abuse-prevention.md`
- **UX Fixes**: See `completed/post-testing-fixes.md`
- **Decorator Bug Fix**: See `completed/decorator-json-fix.md`
- **Current Work**: See `~/.claude/plans/current-work.md`

## Contributing

When completing a feature:
1. Document it in `/docs/completed/`
2. Remove from active plan file
3. Update this README if needed
4. Keep docs concise but complete
