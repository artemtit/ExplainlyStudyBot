# Step 46 — /restart Command

1. Problem being solved in this step: Users may expect a `/restart` command to reset the current flow.
2. Proposed technical solution: Add `/restart` as another alias for returning to the main menu.
3. Implementation plan: Add a `/restart` handler; update help and unknown-command hints; register the command.
4. Example code or pseudocode (if relevant): `/restart -> state.clear(); show_main_menu()`.
5. Risks and edge cases: None; navigation only.
6. Suggested next improvements: Make `/restart` optionally reset the current topic.
