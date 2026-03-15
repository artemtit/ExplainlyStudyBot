# Step 31 — Test Menu Shortcut

1. Problem being solved in this step: While answering test questions, users must navigate back to lesson before reaching the main menu.
2. Proposed technical solution: Add a “Menu” button to the test question keyboard.
3. Implementation plan: Extend `create_test_keyboard` to include a menu button that triggers the existing `test:menu` handler.
4. Example code or pseudocode (if relevant): `keyboard.append([BTN_BACK_MENU -> test:menu])`.
5. Risks and edge cases: None; handler already exists.
6. Suggested next improvements: Add a “Pause” option to save and exit.
