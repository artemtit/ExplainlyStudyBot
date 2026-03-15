# Step 54 — /continue Command

1. Problem being solved in this step: Users need a command to resume their last study session.
2. Proposed technical solution: Add a `/continue` command that triggers the resume flow.
3. Implementation plan: Add a command handler in the study router; call `resume_flow`; update help and bot commands.
4. Example code or pseudocode (if relevant): `/continue -> resume_flow()`.
5. Risks and edge cases: If no resume state exists, show the “no resume” message and return to the menu.
6. Suggested next improvements: Store multiple recent sessions to choose from.
