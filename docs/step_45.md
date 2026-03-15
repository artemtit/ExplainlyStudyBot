# Step 45 — Show Menu After Help

1. Problem being solved in this step: After viewing help, users may want the main menu immediately.
2. Proposed technical solution: Send the main menu after displaying the help text.
3. Implementation plan: Call `show_main_menu` after sending help (command and button).
4. Example code or pseudocode (if relevant): `await message.answer(HELP_TEXT); await show_main_menu(message)`.
5. Risks and edge cases: Slightly more messages; acceptable for clarity.
6. Suggested next improvements: Add a toggle to suppress extra menu message.
