# Step 49 — /settings Command

1. Problem being solved in this step: Users want a direct command to open settings.
2. Proposed technical solution: Add a `/settings` command that opens the settings screen.
3. Implementation plan: Add a command handler in the settings router; update help and unknown-command hints; register the command.
4. Example code or pseudocode (if relevant): `/settings -> _send_settings()`.
5. Risks and edge cases: None; uses existing settings view.
6. Suggested next improvements: Add a settings shortcut in other flows.
