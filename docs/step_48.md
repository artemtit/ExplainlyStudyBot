# Step 48 — /report Command

1. Problem being solved in this step: Users may look for a dedicated command to report issues.
2. Proposed technical solution: Add `/report` as a direct shortcut to the feedback/support channel.
3. Implementation plan: Add a `/report` handler reusing feedback text; update help and unknown-command hints; register the command.
4. Example code or pseudocode (if relevant): `/report -> FEEDBACK_TEXT`.
5. Risks and edge cases: None; it’s an alias.
6. Suggested next improvements: Capture structured bug reports inside the bot.
