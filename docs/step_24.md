# Step 24 — Subscription Placeholder

1. Problem being solved in this step: The “Subscription” button leads to the same profile view, causing confusion.
2. Proposed technical solution: Show a clear “coming soon” message when the subscription button is pressed.
3. Implementation plan: Add a subscription placeholder text and show it on `profile:subscription` callbacks.
4. Example code or pseudocode (if relevant): `edit_or_send(call, SUBSCRIPTION_TEXT)`.
5. Risks and edge cases: None; it’s a passive informational message.
6. Suggested next improvements: Add a waitlist or payment link when ready.
