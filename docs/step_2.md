# Step 2 — User Guidance and Support Entry Point

1. Problem being solved in this step: New users lack an in-bot help entry point and a clear support path when they are stuck or the bot misbehaves.
2. Proposed technical solution: Add a `/help` command and a support button in settings, wired to `SUPPORT_URL` from config.
3. Implementation plan: Extend the start router with a help handler; add a support URL button to the settings keyboard; pass `support_url` through router builders from `bot.main`.
4. Example code or pseudocode (if relevant): `/help` -> send HELP_TEXT + inline button to `SUPPORT_URL`.
5. Risks and edge cases: Missing `SUPPORT_URL` should not break handlers; help should not interfere with ongoing FSM flows.
6. Suggested next improvements: Add a short FAQ for common issues and add a `/feedback` command to collect reports.
