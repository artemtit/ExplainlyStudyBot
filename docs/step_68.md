# Step 68 — /new Alias

1. Problem being solved in this step: Users may try `/new` to start a new topic.
2. Proposed technical solution: Add a `/new` alias that opens the topic entry flow.
3. Implementation plan: Add a command handler in the study router; reuse `show_topic_entry`; update help and command hints.
4. Example code or pseudocode (if relevant): `/new -> show_topic_entry()`.
5. Risks and edge cases: None; same behavior as `/topic`.
6. Suggested next improvements: Offer suggested topics based on recent activity.
