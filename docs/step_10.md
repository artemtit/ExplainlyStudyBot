# Step 10 — Menu Shortcut Command

1. Problem being solved in this step: Users can get lost in flows and need a quick way to return to the main menu.
2. Proposed technical solution: Add a `/menu` command and document it in `/help`.
3. Implementation plan: Add a `/menu` handler that clears state and shows the main menu; extend the help text with `/menu` and `/cancel` descriptions.
4. Example code or pseudocode (if relevant): `/menu` -> `state.clear()` -> `show_main_menu()`.
5. Risks and edge cases: None; handler is idempotent.
6. Suggested next improvements: Add a menu button in inline keyboards for long flows.
