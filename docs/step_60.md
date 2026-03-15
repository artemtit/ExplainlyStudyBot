# Step 60 — /recent Command

1. Problem being solved in this step: Users want a quick way to access their recent topics.
2. Proposed technical solution: Add a `/recent` command that opens the recent-topics picker.
3. Implementation plan: Add a command handler in the study router; reuse `show_topic_entry`; update help and command hints.
4. Example code or pseudocode (if relevant): `/recent -> show_topic_entry()`.
5. Risks and edge cases: If no recent topics exist, show the standard topic prompt.
6. Suggested next improvements: Add a dedicated “recent topics” view with more history.
