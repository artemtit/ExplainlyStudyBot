# Step 3 — Recent Topics Shortcut

1. Problem being solved in this step: Returning users have to retype recently studied topics, slowing down session restarts.
2. Proposed technical solution: Surface the last few topics as quick-select buttons before asking for a new topic.
3. Implementation plan: Add a recent-topics keyboard; show it on “Start learning” if data exists; handle selection via callback and move directly to explanation-level choice.
4. Example code or pseudocode (if relevant): `recent:pick:{idx}` callback -> load topic from state -> show explanation level selector.
5. Risks and edge cases: Empty or stale recent-topic lists; overly long topic labels in buttons.
6. Suggested next improvements: Add “pin topic” or expand the list with pagination; add validation to prevent duplicates.
