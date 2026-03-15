# Step 66 — /me Alias

1. Problem being solved in this step: Users often try `/me` to view their profile.
2. Proposed technical solution: Add a `/me` alias that opens the profile view.
3. Implementation plan: Add a command handler in the profile router; reuse `_open_profile`; update help and command hints.
4. Example code or pseudocode (if relevant): `/me -> _open_profile()`.
5. Risks and edge cases: None; same as `/profile`.
6. Suggested next improvements: Add quick profile actions (settings, support) inline.
