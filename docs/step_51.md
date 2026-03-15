# Step 51 — /tests Command

1. Problem being solved in this step: Users need a direct command to open tests.
2. Proposed technical solution: Add a `/tests` command that starts the test flow.
3. Implementation plan: Add a command handler in the tests router; update help and unknown-command hints; register the command.
4. Example code or pseudocode (if relevant): `/tests -> open_test(reset_progress=True)`.
5. Risks and edge cases: Requires a saved topic; falls back to resume state if available.
6. Suggested next improvements: Add a shortcut for retrying tests from the results screen.
