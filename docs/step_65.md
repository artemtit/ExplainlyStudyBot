# Step 65 — /subscribe Command

1. Problem being solved in this step: Users want a quick way to check subscription status.
2. Proposed technical solution: Add a `/subscribe` command that shows the existing subscription placeholder.
3. Implementation plan: Add a command handler in the profile router; reuse `SUBSCRIPTION_TEXT`; update help and command hints.
4. Example code or pseudocode (if relevant): `/subscribe -> SUBSCRIPTION_TEXT`.
5. Risks and edge cases: None; it’s informational only.
6. Suggested next improvements: Connect subscription to billing and entitlement checks.
