# Step 61 — /streak Command

1. Problem being solved in this step: Users want a quick way to check their daily study streak.
2. Proposed technical solution: Add a `/streak` command that shows the current streak from stats.
3. Implementation plan: Add a command handler in the start router; fetch stats; render a short streak message; register the command.
4. Example code or pseudocode (if relevant): `/streak -> get_user_stats() -> show streak`.
5. Risks and edge cases: Streak may be 0 for new users; show 0 days.
6. Suggested next improvements: Show the best streak or last active day.
