# Step 67 — /resume Alias

1. Problem being solved in this step: Users may try `/resume` instead of `/continue`.
2. Proposed technical solution: Add a `/resume` alias that triggers the resume flow.
3. Implementation plan: Add a command handler in the study router; call `resume_flow`; update help and command hints.
4. Example code or pseudocode (if relevant): `/resume -> resume_flow()`.
5. Risks and edge cases: If no resume state exists, the existing no-resume message is shown.
6. Suggested next improvements: Add a short summary of the resumed topic before opening it.
