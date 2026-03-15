# Step 29 — Flashcards Menu Shortcut

1. Problem being solved in this step: Users in flashcards must go back to the lesson before reaching the main menu.
2. Proposed technical solution: Add a direct “Menu” button to the flashcards keyboard.
3. Implementation plan: Extend `create_flashcards_keyboard` with a menu button and handle `flash:menu` to show the main menu.
4. Example code or pseudocode (if relevant): `flash:menu -> show_main_menu()`.
5. Risks and edge cases: None; this is a simple navigation shortcut.
6. Suggested next improvements: Add similar shortcuts in tests and practice.
