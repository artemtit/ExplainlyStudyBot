# Step 57 Ч /test Alias

1. Problem being solved in this step: Users often try `/test` instead of `/tests`.
2. Proposed technical solution: Add a `/test` alias that запускает тесты.
3. Implementation plan: Add a command handler that calls `open_test`; update help and command hints; register the alias.
4. Example code or pseudocode (if relevant): `/test -> open_test(reset_progress=True)`.
5. Risks and edge cases: None; same flow as `/tests`.
6. Suggested next improvements: Add aliases for other popular commands if needed.
