# Step 12 — Settings Access in Main Menu

1. Problem being solved in this step: Settings are hidden behind the profile screen, adding friction for common actions like progress reset.
2. Proposed technical solution: Add a “Settings” button to the main menu keyboard.
3. Implementation plan: Extend `create_main_menu` to include `BTN_SETTINGS`, reusing existing settings handlers.
4. Example code or pseudocode (if relevant): `keyboard.append([BTN_SETTINGS])`.
5. Risks and edge cases: None; handler already exists.
6. Suggested next improvements: Add a support shortcut in the main menu if needed.
