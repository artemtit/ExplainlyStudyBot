# Step 69 — /main Alias

1. Problem being solved in this step: Users may type `/main` to return to the main menu.
2. Proposed technical solution: Add a `/main` alias that opens the main menu.
3. Implementation plan: Add a command handler in the start router; reuse `show_main_menu`; update help and command hints.
4. Example code or pseudocode (if relevant): `/main -> show_main_menu()`.
5. Risks and edge cases: None; same as `/menu`.
6. Suggested next improvements: Add a short “You are back in the menu” confirmation.
