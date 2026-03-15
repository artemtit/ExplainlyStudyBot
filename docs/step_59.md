# Step 59 — /learn Alias

1. Problem being solved in this step: Some users intuitively try `/learn` to start studying.
2. Proposed technical solution: Add a `/learn` alias that opens the topic entry flow.
3. Implementation plan: Add a command handler in the study router; reuse `show_topic_entry`; update help and command hints.
4. Example code or pseudocode (if relevant): `/learn -> show_topic_entry()`.
5. Risks and edge cases: None; same behavior as `/study` and `/topic`.
6. Suggested next improvements: Consider detecting language preferences to suggest the best command name.
