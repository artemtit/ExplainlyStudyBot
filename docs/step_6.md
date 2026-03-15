# Step 6 — Reset Confirmation

1. Problem being solved in this step: Users can accidentally reset progress with a single tap.
2. Proposed technical solution: Add a confirmation step with explicit “Reset” and “Back” buttons.
3. Implementation plan: Introduce a reset confirmation keyboard; update the settings reset handler to show confirmation; add confirm/cancel callbacks.
4. Example code or pseudocode (if relevant): `settings:reset` -> show confirm keyboard; `settings:reset:confirm` -> perform reset.
5. Risks and edge cases: Callback spam could trigger multiple resets; handlers should be idempotent.
6. Suggested next improvements: Add a brief summary of what will be erased; log reset events.
