# Step 5 — Progress Context

1. Problem being solved in this step: The progress screen shows only totals, which gives little context about the user’s most recent activity.
2. Proposed technical solution: Include the last topic and last stage in the progress summary when available.
3. Implementation plan: Extend `format_progress` to append `last_topic` and `last_stage` with a small stage label map.
4. Example code or pseudocode (if relevant): `if last_topic: add line; if last_stage: map -> label; add line`.
5. Risks and edge cases: Older stats may not have `last_topic`/`last_stage`; keep the extra lines optional.
6. Suggested next improvements: Add last active date and a “Continue” shortcut from progress.
