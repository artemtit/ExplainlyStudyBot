# Step 9 — Continue From Progress

1. Problem being solved in this step: Users viewing progress have no direct shortcut to resume their last session.
2. Proposed technical solution: Add a “Continue” button to the progress screen when a resume state exists.
3. Implementation plan: Extend the progress keyboard with an optional continue button; add a callback to trigger the resume flow.
4. Example code or pseudocode (if relevant): `if load_resume_state: show button; progress:continue -> resume_flow()`.
5. Risks and edge cases: Resume data might be missing by the time the user taps; fallback should show the standard no-resume flow.
6. Suggested next improvements: Add a “Start new topic” button from progress.
