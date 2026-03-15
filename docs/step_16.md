# Step 16 — Friendly Last Active Date

1. Problem being solved in this step: The last active date is shown as raw ISO text, which is not user-friendly.
2. Proposed technical solution: Format ISO dates into a readable Russian date string (e.g., “15 марта 2026”).
3. Implementation plan: Add a small ISO date formatter and use it in `format_progress` when `last_active_date` is present.
4. Example code or pseudocode (if relevant): `format_date_iso('2026-03-15T...') -> '15 марта 2026'`.
5. Risks and edge cases: Non-ISO values should fall back to the raw string.
6. Suggested next improvements: Add time of day and local timezone if needed.
