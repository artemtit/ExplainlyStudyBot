# Step 27 — Back From Explanation Level

1. Problem being solved in this step: Users cannot easily return to topic entry after opening explanation level selection.
2. Proposed technical solution: Add a “Back” button to the explanation level keyboard.
3. Implementation plan: Extend the keyboard with a back button and handle `explain_level:back` by returning to the topic-entry flow.
4. Example code or pseudocode (if relevant): `explain_level:back -> show_topic_entry()`.
5. Risks and edge cases: None; state transitions are already supported.
6. Suggested next improvements: Add a “cancel” button during long-running steps.
