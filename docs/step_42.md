# Step 42 — Review Lesson After Test

1. Problem being solved in this step: After finishing a test, users may want to return to the lesson quickly.
2. Proposed technical solution: Add a “Review lesson” button to the test completion keyboard.
3. Implementation plan: Add `BTN_REVIEW_LESSON` to `create_test_done_keyboard` (handler already exists).
4. Example code or pseudocode (if relevant): `test:review -> render_lesson()`.
5. Risks and edge cases: None; uses existing handler.
6. Suggested next improvements: Add a short summary of mistakes before returning.
