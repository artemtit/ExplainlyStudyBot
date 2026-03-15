# Step 55 — /profile Command

1. Problem being solved in this step: Users want a direct command to view their profile.
2. Proposed technical solution: Add a `/profile` command that opens the profile screen.
3. Implementation plan: Add a command handler in the profile router; update help and unknown-command hints; register the command.
4. Example code or pseudocode (if relevant): `/profile -> _open_profile()`.
5. Risks and edge cases: None; profile already handles missing data gracefully.
6. Suggested next improvements: Add profile settings shortcuts inside the profile view.
