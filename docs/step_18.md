# Step 18 — Restart Flashcards

1. Problem being solved in this step: Users have to back out and re-enter flashcards to start over.
2. Proposed technical solution: Add a “Restart” button in flashcards to reset to the first card.
3. Implementation plan: Add a restart button in the flashcards keyboard; handle a `flash:restart` callback to reset `card_index` and re-render.
4. Example code or pseudocode (if relevant): `on restart -> card_index=0; flash_show_answer=False; render`.
5. Risks and edge cases: None; should be safe in any flashcards state.
6. Suggested next improvements: Add a shuffle option.
