# Step 58 — /flashcards Alias

1. Problem being solved in this step: Users may try the full `/flashcards` command instead of `/cards`.
2. Proposed technical solution: Add a `/flashcards` alias that opens the flashcards flow.
3. Implementation plan: Add a command handler in the flashcards router; update help and unknown-command hints; register the alias.
4. Example code or pseudocode (if relevant): `/flashcards -> open_flashcards(reset_index=True)`.
5. Risks and edge cases: None; same flow as `/cards`.
6. Suggested next improvements: Consider `/cards` and `/flashcards` autocomplete in the menu.
