# Step 33 — /stats Alias

1. Problem being solved in this step: Some users expect a `/stats` command instead of `/progress`.
2. Proposed technical solution: Add `/stats` as an alias for the progress screen.
3. Implementation plan: Add a `/stats` handler reusing the progress logic; update help and unknown-command hints; register the command with Telegram.
4. Example code or pseudocode (if relevant): `/stats` -> same as `/progress`.
5. Risks and edge cases: None; read-only.
6. Suggested next improvements: Add `/stats` to quick reply menus if needed.
