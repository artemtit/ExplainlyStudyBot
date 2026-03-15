# Step 40 — Menu in Lesson Actions

1. Problem being solved in this step: From the lesson action keyboard, users cannot jump directly to the main menu.
2. Proposed technical solution: Add a “Menu” button to the lesson keyboard.
3. Implementation plan: Extend `create_lesson_keyboard` with `lesson:menu` and handle it in the study router.
4. Example code or pseudocode (if relevant): `lesson:menu -> show_main_menu()`.
5. Risks and edge cases: None; navigation only.
6. Suggested next improvements: Add a “Help” button to lesson actions.
