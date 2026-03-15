# Step 63 — /reset Command

1. Problem being solved in this step: Users want a direct way to reset their progress without digging into settings.
2. Proposed technical solution: Add a `/reset` command that opens the reset confirmation screen.
3. Implementation plan: Add a command handler in the settings router; show the existing reset confirmation UI; update help and command hints.
4. Example code or pseudocode (if relevant): `/reset -> RESET_CONFIRM_TEXT + confirm keyboard`.
5. Risks and edge cases: Destructive action; ensure it always shows a confirmation step.
6. Suggested next improvements: Add a short explanation of what gets reset.
