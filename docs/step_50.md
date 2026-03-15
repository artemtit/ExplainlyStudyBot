# Step 50 — /cards Command

1. Problem being solved in this step: Users want a direct command to open flashcards.
2. Proposed technical solution: Add a `/cards` command that opens the flashcards flow.
3. Implementation plan: Add a command handler in the flashcards router; update help and unknown-command hints; register the command.
4. Example code or pseudocode (if relevant): `/cards -> open_flashcards(reset_index=True)`.
5. Risks and edge cases: Requires a recent topic; falls back to resume state.
6. Suggested next improvements: Add an alias for `/cards` in menu shortcuts.
