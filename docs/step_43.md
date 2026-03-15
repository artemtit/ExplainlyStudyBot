# Step 43 — /home Alias

1. Problem being solved in this step: Users may expect `/home` to return to the main menu.
2. Proposed technical solution: Add `/home` as an alias for `/menu`.
3. Implementation plan: Add a `/home` handler, update help and unknown-command hints, and register the command.
4. Example code or pseudocode (if relevant): `/home -> state.clear(); show_main_menu()`.
5. Risks and edge cases: None; navigation only.
6. Suggested next improvements: Add aliases for `/start` if needed.
