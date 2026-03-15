# Step 52 — /practice Command

1. Problem being solved in this step: Users want a quick command to open practice tasks.
2. Proposed technical solution: Add a `/practice` command that opens the practice flow.
3. Implementation plan: Add a command handler in the tests router; update help and unknown-command hints; register the command.
4. Example code or pseudocode (if relevant): `/practice -> open_practice(show_solution=False)`.
5. Risks and edge cases: Requires a recent topic; falls back to resume state when available.
6. Suggested next improvements: Allow switching between practice and tests without returning to the menu.
