# Step 21 — Telegram Command Menu

1. Problem being solved in this step: Users may not know which commands are available in the bot.
2. Proposed technical solution: Register bot commands with Telegram so they appear in the client UI.
3. Implementation plan: Call `set_my_commands` during startup with the supported commands.
4. Example code or pseudocode (if relevant): `await bot.set_my_commands([...])`.
5. Risks and edge cases: API failures should not block startup; handle errors gracefully.
6. Suggested next improvements: Provide locale-specific command descriptions.
