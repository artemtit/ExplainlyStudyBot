# Step 70 — /topics Command

1. Problem being solved in this step: Users want a quick list of their recent topics without starting a new flow.
2. Proposed technical solution: Add a `/topics` command that shows a larger recent-topics list.
3. Implementation plan: Add a command handler in the study router; reuse the recent-topics UI with a higher limit; update help and command hints.
4. Example code or pseudocode (if relevant): `/topics -> show_recent_topics(limit=10)`.
5. Risks and edge cases: If no recent topics exist, fall back to the standard topic prompt.
6. Suggested next improvements: Add search or filtering for long topic histories.
