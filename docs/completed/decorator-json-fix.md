# Authentication Decorator JSON Fix (Issue #13)

**Status:** ✅ Complete

## Problem
Frontend JavaScript calling professor API routes received HTML error pages instead of JSON, causing:
```
Error: Unexpected token '<', "<!doctype "... is not valid JSON
```

## Root Cause
Decorators `@professor_required` and `@professor_owns_ta` always returned HTML redirects on auth failure, even for AJAX/API requests.

## Solution
Made decorators content-aware - return JSON for API calls, HTML for page requests:

```python
if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
    return jsonify({"error": "Access denied"}), 403
```

## Files Modified
- **professor.py** (lines 28-50) - Both decorators updated

## Impact
Fixed ALL professor API routes:
- Document upload/delete/update
- TA indexing
- Pause/resume
- Delete TA
- Indexing status polling

## Testing
✅ Document indexing now works
✅ Auth failures return proper JSON errors
✅ No more HTML parsing errors
