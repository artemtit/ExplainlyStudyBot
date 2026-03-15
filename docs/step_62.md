# Step 62 — /last Command

1. Problem being solved in this step: Users want to quickly see their last studied topic and mode.
2. Proposed technical solution: Add a `/last` command that reads the resume state and shows the last topic/stage.
3. Implementation plan: Add a command handler in the start router; load resume state; render a short summary; fall back to a no-resume message.
4. Example code or pseudocode (if relevant): `/last -> load_resume_state() -> show last topic`.
5. Risks and edge cases: If no resume data exists, show a helpful message.
6. Suggested next improvements: Offer a quick “continue” button from this screen.
