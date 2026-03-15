# Step 53 — /lesson Command

1. Problem being solved in this step: Users want a direct command to reopen the current lesson.
2. Proposed technical solution: Add a `/lesson` command that renders the lesson view for the active or last topic.
3. Implementation plan: Add a command handler in the study router; load material from state or resume data; render the lesson; update help and commands.
4. Example code or pseudocode (if relevant): `if material: render_lesson(); else load resume -> render_lesson()`.
5. Risks and edge cases: If no resume data exists, show the “no resume” message and return to the menu.
6. Suggested next improvements: Add a `/lesson` shortcut from test results to jump back to the lesson.
