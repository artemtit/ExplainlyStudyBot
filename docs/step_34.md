# Step 34 — Help Button in Main Menu

1. Problem being solved in this step: Users may not know or use the `/help` command.
2. Proposed technical solution: Add a “Help” button to the main menu keyboard that shows the help text.
3. Implementation plan: Add `BTN_HELP` to the main menu and handle it in the start router.
4. Example code or pseudocode (if relevant): `F.text == BTN_HELP -> send HELP_TEXT`.
5. Risks and edge cases: None; it is read-only.
6. Suggested next improvements: Add a “Support” button in the main menu.
