# Step 20 — /progress Command

1. Problem being solved in this step: Users can only access progress via the menu button, which is slower for power users.
2. Proposed technical solution: Add a `/progress` command that shows the progress screen.
3. Implementation plan: Extend the start router with a progress handler; reuse `format_progress` and `create_progress_keyboard`; list the command in `/help` and unknown-command hints.
4. Example code or pseudocode (if relevant): `/progress` -> fetch stats + resume -> render progress.
5. Risks and edge cases: None; handler is read-only.
6. Suggested next improvements: Add `/stats` alias if desired.
