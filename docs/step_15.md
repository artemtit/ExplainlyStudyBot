# Step 15 — Last Active Date in Progress

1. Problem being solved in this step: Progress screen lacks context about when the user last studied.
2. Proposed technical solution: Display the last active date if available.
3. Implementation plan: Extend `format_progress` to append `last_active_date` when present.
4. Example code or pseudocode (if relevant): `if last_active: add line`.
5. Risks and edge cases: Older records may not have `last_active_date`; keep the line optional.
6. Suggested next improvements: Format the date in a more user-friendly locale string.
