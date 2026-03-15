# Step 17 — New Topic From Progress

1. Problem being solved in this step: From the progress screen, users can only go back or continue, not start a fresh topic.
2. Proposed technical solution: Add a “New topic” action in progress that opens the topic entry flow (including recent topics).
3. Implementation plan: Add a new progress keyboard button and callback; reuse a shared topic-entry helper used by the main “Start learning” flow.
4. Example code or pseudocode (if relevant): `progress:new_topic -> show_topic_entry()`.
5. Risks and edge cases: None; handler clears to the topic-entry state.
6. Suggested next improvements: Add a shortcut to pick from popular topics.
