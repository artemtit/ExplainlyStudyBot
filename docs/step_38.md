# Step 38 — /about Command

1. Problem being solved in this step: Users may want a short description of the bot’s purpose.
2. Proposed technical solution: Add an `/about` command with a concise description.
3. Implementation plan: Add `ABOUT_TEXT`, handle `/about`, update help and unknown-command hints, register the command in Telegram.
4. Example code or pseudocode (if relevant): `/about -> send ABOUT_TEXT`.
5. Risks and edge cases: None; static text.
6. Suggested next improvements: Add version/build info.
