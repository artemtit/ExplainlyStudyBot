# Step 14 — Support Command

1. Problem being solved in this step: Users need a direct command to contact support without navigating menus.
2. Proposed technical solution: Add a `/support` command that shares the support link when configured.
3. Implementation plan: Extend the start router with a support handler; reuse the support inline button; add `/support` to help and unknown-command text.
4. Example code or pseudocode (if relevant): `if support_url: send SUPPORT_TEXT + button else show missing message`.
5. Risks and edge cases: Support URL may be unset; show a graceful fallback.
6. Suggested next improvements: Add a feedback flow that captures a short message before redirecting.
