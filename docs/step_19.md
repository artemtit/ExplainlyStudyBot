# Step 19 — Shuffle Flashcards

1. Problem being solved in this step: Flashcards are always presented in the same order, reducing variety for repeated review.
2. Proposed technical solution: Add a “Shuffle” button that randomizes card order for the current session.
3. Implementation plan: Add a shuffle button to the flashcards keyboard; on shuffle, store a randomized order in state and render using that order.
4. Example code or pseudocode (if relevant): `order = shuffle(range(len(cards)))` -> `flash_order` -> map cards by order.
5. Risks and edge cases: Ensure indices remain valid if cards length changes; reset order on new session.
6. Suggested next improvements: Add a toggle to persist shuffle preference.
