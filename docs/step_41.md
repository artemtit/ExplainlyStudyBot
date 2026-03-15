# Step 41 — Menu in Test Results

1. Problem being solved in this step: After answering a test question, users cannot jump straight to the main menu.
2. Proposed technical solution: Add a “Menu” button to the test result keyboard.
3. Implementation plan: Extend `create_test_result_keyboard` with `test:menu` (already handled).
4. Example code or pseudocode (if relevant): `keyboard.append([BTN_BACK_MENU -> test:menu])`.
5. Risks and edge cases: None; navigation only.
6. Suggested next improvements: Add a “Retry question” option.
