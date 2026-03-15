# Step 25 — Retry Test

1. Problem being solved in this step: Users cannot repeat the same test without generating a new one.
2. Proposed technical solution: Add a “Retry test” button to restart the current test with the same questions.
3. Implementation plan: Add a new button to the test completion keyboard and handle `test:retry` by resetting test progress.
4. Example code or pseudocode (if relevant): `test:retry -> open_test(reset_progress=True)`.
5. Risks and edge cases: None; the test list is already in state.
6. Suggested next improvements: Allow retry from the review screen as well.
