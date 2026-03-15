# Step 56 — /study Command

1. Problem being solved in this step: Users want a clear command to start a new study topic.
2. Proposed technical solution: Add a `/study` alias that opens the topic entry flow.
3. Implementation plan: Add a command handler in the study router; reuse `show_topic_entry`; update help and command hints.
4. Example code or pseudocode (if relevant): `/study -> show_topic_entry()`.
5. Risks and edge cases: None; same behavior as the “Start learning” button.
6. Suggested next improvements: Add a quick “recent topics” picker for `/study`.
