# Step 39 — /feedback Command

1. Problem being solved in this step: Users have no clear command to send feedback.
2. Proposed technical solution: Add a `/feedback` command that points to support.
3. Implementation plan: Add feedback text and handler; update help and unknown-command hints; register command with Telegram.
4. Example code or pseudocode (if relevant): `/feedback -> send FEEDBACK_TEXT + support button`.
5. Risks and edge cases: None; static text.
6. Suggested next improvements: Add a feedback collection flow inside the bot.
