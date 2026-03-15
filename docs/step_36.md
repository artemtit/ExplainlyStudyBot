# Step 36 — Menu in Explanation Level

1. Problem being solved in this step: From explanation level selection, users cannot jump directly to the main menu.
2. Proposed technical solution: Add a “Menu” button to the explanation level keyboard.
3. Implementation plan: Extend `create_explanation_level_keyboard` with a menu button and handle `explain_level:menu` to show the main menu.
4. Example code or pseudocode (if relevant): `explain_level:menu -> show_main_menu()`.
5. Risks and edge cases: None; navigation only.
6. Suggested next improvements: Add a “Help” button to this screen.
